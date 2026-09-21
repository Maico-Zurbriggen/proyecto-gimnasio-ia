from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class ClaimedGeneration:
    request_id: UUID
    attempt_id: UUID
    attempt_number: int
    minimized_context: dict[str, Any]
    preferences: dict[str, Any]
    retention_until: datetime


class GenerationRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def ping(self) -> None:
        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            await connection.execute("SELECT 1")

    async def request_exists(self, request_id: UUID) -> bool:
        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            cursor = await connection.execute(
                """
                SELECT 1
                FROM ai_integration.ai_generation_requests
                WHERE id = %s
                  AND state IN ('PENDIENTE', 'PROCESANDO')
                  AND retention_until > now()
                """,
                (request_id,),
            )
            return await cursor.fetchone() is not None

    async def claim(
        self,
        request_id: UUID,
        worker_id: str,
        lease_seconds: int,
        contract_version: str,
        model_version: str,
        configuration_version: str,
    ) -> ClaimedGeneration | None:
        async with await psycopg.AsyncConnection.connect(
            self._database_url, row_factory=dict_row
        ) as connection, connection.transaction():
            cursor = await connection.execute(
                """
                    SELECT id, minimized_context, preferences, retention_until
                    FROM ai_integration.ai_generation_requests
                    WHERE id = %s
                      AND available_at <= now()
                      AND retention_until > now()
                      AND (
                        state = 'PENDIENTE'
                        OR (state = 'PROCESANDO' AND lease_until < now())
                      )
                    FOR UPDATE SKIP LOCKED
                    """,
                (request_id,),
            )
            request = await cursor.fetchone()
            if request is None:
                return None

            attempt_cursor = await connection.execute(
                """
                    SELECT COALESCE(MAX(attempt_number), 0) + 1 AS attempt_number
                    FROM ai_integration.ai_generation_attempts
                    WHERE request_id = %s
                    """,
                (request_id,),
            )
            attempt_row = await attempt_cursor.fetchone()
            if attempt_row is None:
                raise RuntimeError("Could not allocate a generation attempt")
            attempt_number = int(attempt_row["attempt_number"])
            input_hash = _hash_input(
                request["minimized_context"], request["preferences"]
            )
            lease_until = datetime.now(UTC) + timedelta(seconds=lease_seconds)

            await connection.execute(
                """
                    UPDATE ai_integration.ai_generation_requests
                    SET state = 'PROCESANDO', lease_owner = %s, lease_until = %s
                    WHERE id = %s
                    """,
                (worker_id, lease_until, request_id),
            )
            created = await connection.execute(
                """
                    INSERT INTO ai_integration.ai_generation_attempts (
                      request_id, attempt_number, state, started_at,
                      contract_version, input_hash, model_version,
                      configuration_version
                    )
                    VALUES (%s, %s, 'PROCESANDO', now(), %s, %s, %s, %s)
                    RETURNING id
                    """,
                (
                    request_id,
                    attempt_number,
                    contract_version,
                    input_hash,
                    model_version,
                    configuration_version,
                ),
            )
            created_attempt = await created.fetchone()
            if created_attempt is None:
                raise RuntimeError("Could not create a generation attempt")
            attempt_id = created_attempt["id"]

            return ClaimedGeneration(
                request_id=request_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                minimized_context=request["minimized_context"],
                preferences=request["preferences"],
                retention_until=request["retention_until"],
            )

    async def complete(
        self,
        claimed: ClaimedGeneration,
        output: dict[str, Any],
        output_hash: str,
        model_version: str,
        configuration_version: str,
    ) -> None:
        async with await psycopg.AsyncConnection.connect(
            self._database_url
        ) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO ai_integration.ai_generation_results (
                  attempt_id, structured_output, output_hash,
                  structurally_valid, retention_until
                )
                VALUES (%s, %s::jsonb, %s, true, %s)
                """,
                (
                    claimed.attempt_id,
                    json.dumps(output, ensure_ascii=False),
                    output_hash,
                    claimed.retention_until,
                ),
            )
            await connection.execute(
                """
                UPDATE ai_integration.ai_generation_attempts
                SET state = 'COMPLETADO', finished_at = now(),
                    model_version = %s, configuration_version = %s
                WHERE id = %s
                """,
                (model_version, configuration_version, claimed.attempt_id),
            )
            await connection.execute(
                """
                UPDATE ai_integration.ai_generation_requests
                SET state = 'COMPLETADA', finished_at = now(),
                    lease_owner = NULL, lease_until = NULL
                WHERE id = %s
                """,
                (claimed.request_id,),
            )

    async def fail(
        self,
        claimed: ClaimedGeneration,
        attempt_state: str,
        error_code: str,
        max_attempts: int,
    ) -> bool:
        exhausted = claimed.attempt_number >= max_attempts
        request_state = "NO_DISPONIBLE" if exhausted else "PENDIENTE"
        async with await psycopg.AsyncConnection.connect(
            self._database_url
        ) as connection, connection.transaction():
            await connection.execute(
                """
                UPDATE ai_integration.ai_generation_attempts
                SET state = %s::ai_integration."AiGenerationAttemptState",
                    finished_at = now(), error_code = %s
                WHERE id = %s
                """,
                (attempt_state, error_code, claimed.attempt_id),
            )
            await connection.execute(
                """
                UPDATE ai_integration.ai_generation_requests
                SET state = %s::ai_integration."AiGenerationRequestState",
                    available_at = CASE WHEN %s THEN available_at ELSE now() END,
                    finished_at = CASE WHEN %s THEN now() ELSE NULL END,
                    lease_owner = NULL, lease_until = NULL
                WHERE id = %s
                """,
                (request_state, exhausted, exhausted, claimed.request_id),
            )
        return exhausted


def _hash_input(context: dict[str, Any], preferences: dict[str, Any]) -> str:
    raw = json.dumps(
        {"minimized_context": context, "preferences": preferences},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()
