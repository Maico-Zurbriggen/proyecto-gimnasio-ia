from __future__ import annotations

import asyncio
import os
import selectors
import sys
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest

from gym_engine.persistence import GenerationRepository


@pytest.mark.parametrize(
    "error_code",
    [
        "gym_warmup_sets_outside_settings",
        "routine_regeneration_candidate_unchanged",
    ],
)
def test_validation_failure_survives_worker_restart_for_specific_retry_feedback(
    error_code: str,
) -> None:
    database_url = os.getenv("AI_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Requires isolated AI_TEST_DATABASE_URL")
    target = urlparse(database_url)
    if (target.hostname, target.port, target.path) != ("127.0.0.1", 65433, "/gym_renewal_qa"):
        pytest.fail("Refusing a database other than the isolated renewal QA instance")
    # psycopg receives the connection URL without Prisma's schema option.
    database_url = database_url.split("?", 1)[0]
    identifier = uuid4()

    async def verify() -> None:
        async with await psycopg.AsyncConnection.connect(database_url) as connection:
            await connection.execute(
                """INSERT INTO ai_integration.ai_generation_requests
                (id, idempotency_key, minimized_context, preferences, context_hash, retention_until)
                VALUES (%s, %s, '{}'::jsonb, '{"schema_version":"2.0"}'::jsonb,
                        'synthetic-policy-retry', now() + interval '1 day')""",
                (identifier, str(identifier)),
            )
        try:
            arguments = dict(
                request_id=identifier,
                worker_id="qa-only",
                lease_seconds=30,
                contract_version="routine-generation@2.0",
                model_version="qa-model",
                configuration_version="qa-prompt",
                max_attempts=2,
            )
            repository = GenerationRepository(database_url)
            first = await repository.claim(**arguments)
            assert first is not None
            assert first.previous_error_code is None
            await repository.fail(
                first,
                attempt_state="SALIDA_INVALIDA",
                error_code=error_code,
                max_attempts=2,
            )
            async with await psycopg.AsyncConnection.connect(database_url) as connection:
                await connection.execute(
                    "UPDATE ai_integration.ai_generation_requests "
                    "SET available_at=now() WHERE id=%s",
                    (identifier,),
                )
            # A different instance gets the correction from durable storage, not memory.
            second = await GenerationRepository(database_url).claim(**arguments)
            assert second is not None
            assert second.attempt_number == 2
            assert second.previous_error_code == error_code
        finally:
            async with await psycopg.AsyncConnection.connect(database_url) as connection:
                await connection.execute(
                    "DELETE FROM ai_integration.ai_generation_attempts WHERE request_id=%s",
                    (identifier,),
                )
                await connection.execute(
                    "DELETE FROM ai_integration.ai_generation_requests WHERE id=%s",
                    (identifier,),
                )

    if sys.platform == "win32":
        asyncio.run(
            verify(), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        )
    else:
        asyncio.run(verify())
