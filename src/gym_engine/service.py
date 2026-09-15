from __future__ import annotations

import socket
from uuid import UUID

import httpx
from pydantic import ValidationError

from .config import Settings
from .llm import OllamaClient
from .persistence import GenerationRepository


class GenerationProcessingError(RuntimeError):
    pass


async def process_generation(request_id: UUID, settings: Settings) -> None:
    repository = GenerationRepository(settings.database_url)
    claimed = await repository.claim(
        request_id=request_id,
        worker_id=f"vercel:{socket.gethostname()}",
        lease_seconds=settings.timeout_seconds + 30,
        contract_version="routine-generation@1.0",
        model_version=settings.llm_model,
        configuration_version=settings.configuration_version,
    )
    if claimed is None:
        return

    client = OllamaClient(
        base_url=settings.llm_api_url,
        model=settings.llm_model,
        api_token=settings.llm_api_token,
        timeout_seconds=settings.timeout_seconds,
    )
    try:
        result = await client.generate_routine(
            claimed.minimized_context, claimed.preferences
        )
        await repository.complete(
            claimed=claimed,
            output=result.output.model_dump(mode="json"),
            output_hash=result.output_hash,
            model_version=result.model_version,
            configuration_version=settings.configuration_version,
        )
    except httpx.TimeoutException as error:
        exhausted = await repository.fail(
            claimed,
            attempt_state="AGOTADO_POR_TIEMPO",
            error_code="llm_timeout",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM timed out") from error
    except ValidationError as error:
        exhausted = await repository.fail(
            claimed,
            attempt_state="SALIDA_INVALIDA",
            error_code="invalid_structured_output",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM returned invalid output") from error
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        exhausted = await repository.fail(
            claimed,
            attempt_state="FALLIDO",
            error_code="llm_request_failed",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM request failed") from error
