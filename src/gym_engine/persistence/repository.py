"""Acceso a ai_integration.ai_generation_* (esquema real, verificado contra Neon test).

El claim es el mismo patrón que ya está probado en develop (src/gym_engine/persistence.py,
async/psycopg): reclamo atómico con FOR UPDATE SKIP LOCKED + lease, una fila de intento
por cada llamada al LLM. Acá va en sync porque el resto de esta rama (API, poller) es sync;
la lógica SQL es la misma.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb

from gym_engine.persistence.models import AttemptState, ClaimedGeneration, RequestStatus


def _hash_input(minimized_context: dict[str, Any], preferences: dict[str, Any]) -> str:
    raw = json.dumps(
        {"minimized_context": minimized_context, "preferences": preferences},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def get_by_idempotency_key(
    conn: psycopg.Connection[DictRow], idempotency_key: str
) -> DictRow | None:
    cur = conn.execute(
        """
        select id, state
        from ai_integration.ai_generation_requests
        where idempotency_key = %s
        """,
        (idempotency_key,),
    )
    return cur.fetchone()


def create_request(
    conn: psycopg.Connection[DictRow],
    *,
    idempotency_key: str,
    minimized_context: dict[str, Any],
    preferences: dict[str, Any],
    retention_days: int,
) -> DictRow:
    context_hash = _hash_input(minimized_context, preferences)
    retention_until = datetime.now(UTC) + timedelta(days=retention_days)
    cur = conn.execute(
        """
        insert into ai_integration.ai_generation_requests (
          id, idempotency_key, state, minimized_context, preferences, context_hash,
          available_at, created_at, retention_until
        )
        values (%s, %s, 'PENDIENTE', %s, %s, %s, now(), now(), %s)
        returning id, state
        """,
        (
            uuid.uuid4(),
            idempotency_key,
            Jsonb(minimized_context),
            Jsonb(preferences),
            context_hash,
            retention_until,
        ),
    )
    row = cur.fetchone()
    assert row is not None
    return row


def get_status(conn: psycopg.Connection[DictRow], request_id: uuid.UUID) -> RequestStatus | None:
    cur = conn.execute(
        """
        select id, state
        from ai_integration.ai_generation_requests
        where id = %s
        """,
        (request_id,),
    )
    request = cur.fetchone()
    if request is None:
        return None

    result_cur = conn.execute(
        """
        select r.structured_output
        from ai_integration.ai_generation_results r
        join ai_integration.ai_generation_attempts a on a.id = r.attempt_id
        where a.request_id = %s
        order by r.created_at desc
        limit 1
        """,
        (request_id,),
    )
    result = result_cur.fetchone()

    error_cur = conn.execute(
        """
        select error_code
        from ai_integration.ai_generation_attempts
        where request_id = %s and error_code is not null
        order by attempt_number desc
        limit 1
        """,
        (request_id,),
    )
    error = error_cur.fetchone()

    return RequestStatus(
        request_id=request["id"],
        state=request["state"],
        structured_output=result["structured_output"] if result else None,
        error_code=error["error_code"] if error else None,
    )


def claim_next_pending(
    conn: psycopg.Connection[DictRow],
    *,
    worker_id: str,
    lease_seconds: int,
    contract_version: str,
    model_version: str,
    configuration_version: str,
) -> ClaimedGeneration | None:
    """Reclamo atómico: FOR UPDATE SKIP LOCKED + lease, crea la fila de intento."""
    cur = conn.execute(
        """
        select id, minimized_context, preferences, retention_until
        from ai_integration.ai_generation_requests
        where available_at <= now()
          and retention_until > now()
          and (
            state = 'PENDIENTE'
            or (state = 'PROCESANDO' and lease_until < now())
          )
        order by available_at
        limit 1
        for update skip locked
        """
    )
    request = cur.fetchone()
    if request is None:
        return None

    request_id = request["id"]
    attempt_cur = conn.execute(
        """
        select coalesce(max(attempt_number), 0) + 1 as attempt_number
        from ai_integration.ai_generation_attempts
        where request_id = %s
        """,
        (request_id,),
    )
    attempt_row = attempt_cur.fetchone()
    assert attempt_row is not None
    attempt_number = int(attempt_row["attempt_number"])
    input_hash = _hash_input(request["minimized_context"], request["preferences"])
    lease_until = datetime.now(UTC) + timedelta(seconds=lease_seconds)

    conn.execute(
        """
        update ai_integration.ai_generation_requests
        set state = 'PROCESANDO', lease_owner = %s, lease_until = %s
        where id = %s
        """,
        (worker_id, lease_until, request_id),
    )
    created = conn.execute(
        """
        insert into ai_integration.ai_generation_attempts (
          id, request_id, attempt_number, state, started_at,
          contract_version, input_hash, model_version, configuration_version
        )
        values (%s, %s, %s, 'PROCESANDO', now(), %s, %s, %s, %s)
        returning id
        """,
        (
            uuid.uuid4(),
            request_id,
            attempt_number,
            contract_version,
            input_hash,
            model_version,
            configuration_version,
        ),
    )
    created_row = created.fetchone()
    assert created_row is not None

    return ClaimedGeneration(
        request_id=request_id,
        attempt_id=created_row["id"],
        attempt_number=attempt_number,
        minimized_context=request["minimized_context"],
        preferences=request["preferences"],
        retention_until=request["retention_until"],
    )


def complete(
    conn: psycopg.Connection[DictRow],
    claimed: ClaimedGeneration,
    *,
    output: dict[str, Any],
    output_hash: str,
    model_version: str,
    configuration_version: str,
) -> None:
    conn.execute(
        """
        insert into ai_integration.ai_generation_results (
          id, attempt_id, structured_output, output_hash, structurally_valid,
          created_at, retention_until
        )
        values (%s, %s, %s, %s, true, now(), %s)
        """,
        (
            uuid.uuid4(),
            claimed.attempt_id,
            Jsonb(output),
            output_hash,
            claimed.retention_until,
        ),
    )
    conn.execute(
        """
        update ai_integration.ai_generation_attempts
        set state = 'COMPLETADO', finished_at = now(),
            model_version = %s, configuration_version = %s
        where id = %s
        """,
        (model_version, configuration_version, claimed.attempt_id),
    )
    conn.execute(
        """
        update ai_integration.ai_generation_requests
        set state = 'COMPLETADA', finished_at = now(), lease_owner = null, lease_until = null
        where id = %s
        """,
        (claimed.request_id,),
    )


def fail(
    conn: psycopg.Connection[DictRow],
    claimed: ClaimedGeneration,
    *,
    attempt_state: AttemptState,
    error_code: str,
    max_attempts: int,
) -> bool:
    """Marca el intento fallido. Devuelve True si se agotaron los reintentos."""
    exhausted = claimed.attempt_number >= max_attempts
    request_state = "NO_DISPONIBLE" if exhausted else "PENDIENTE"
    conn.execute(
        """
        update ai_integration.ai_generation_attempts
        set state = %s::ai_integration."AiGenerationAttemptState",
            finished_at = now(), error_code = %s
        where id = %s
        """,
        (attempt_state, error_code, claimed.attempt_id),
    )
    conn.execute(
        """
        update ai_integration.ai_generation_requests
        set state = %s::ai_integration."AiGenerationRequestState",
            available_at = case when %s then available_at else now() end,
            finished_at = case when %s then now() else null end,
            lease_owner = null, lease_until = null
        where id = %s
        """,
        (request_state, exhausted, exhausted, claimed.request_id),
    )
    return exhausted
