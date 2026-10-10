from __future__ import annotations

import asyncio
import socket
from logging import getLogger
from uuid import UUID

import httpx
from pydantic import ValidationError

from .config import Settings
from .generation_prompts import VALIDATION_CORRECTIONS
from .llm import InvalidPrescriptionError, OllamaClient
from .llm_v2 import ContextCapacityError, configuration_signature
from .persistence import GenerationRepository

logger = getLogger(__name__)


class GenerationProcessingError(RuntimeError):
    pass


async def process_generation(request_id: UUID, settings: Settings) -> None:
    repository = GenerationRepository(settings.database_url)
    configuration = configuration_signature(
        settings.configuration_version, settings.context_tokens, settings.response_tokens
    )
    claimed = await repository.claim(
        request_id=request_id,
        worker_id=f"{settings.generation_queue_mode}:{socket.gethostname()}",
        lease_seconds=settings.timeout_seconds + 30,
        contract_version="routine-generation@1.0",
        model_version=settings.llm_model,
        configuration_version=configuration,
        max_attempts=settings.max_attempts,
    )
    if claimed is None:
        return

    client = OllamaClient(
        base_url=settings.llm_api_url,
        model=settings.llm_model,
        api_token=settings.llm_api_token,
        timeout_seconds=settings.timeout_seconds,
        context_tokens=settings.context_tokens,
        response_tokens=settings.response_tokens,
    )
    try:
        result = await client.generate_routine(
            claimed.minimized_context,
            claimed.preferences,
            retry=claimed.attempt_number > 1,
            validation_feedback=claimed.previous_error_code,
        )
        await repository.complete(
            claimed=claimed,
            output=result.output.model_dump(mode="json"),
            output_hash=result.output_hash,
            model_version=result.model_version,
            configuration_version=configuration,
        )
    except ContextCapacityError:
        await repository.fail(
            claimed,
            attempt_state="FALLIDO",
            error_code="context_capacity_exceeded",
            max_attempts=claimed.attempt_number,
        )
    except asyncio.CancelledError:
        await repository.fail(
            claimed,
            attempt_state="FALLIDO",
            error_code="worker_interrupted",
            max_attempts=settings.max_attempts,
        )
        raise
    except (httpx.TimeoutException, TimeoutError) as error:
        logger.warning("Generation request %s timed out", request_id)
        exhausted = await repository.fail(
            claimed,
            attempt_state="AGOTADO_POR_TIEMPO",
            error_code="llm_timeout",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM timed out") from error
    except (ValidationError, InvalidPrescriptionError) as error:
        detail = (
            "; ".join(
                f"{item['loc']}: {item['type']}"
                for item in error.errors(
                    include_input=False, include_context=False, include_url=False
                )
            )
            if isinstance(error, ValidationError)
            else str(error)
        )
        logger.warning(
            "Generation request %s returned invalid structured output (%s)", request_id, detail
        )
        exhausted = await repository.fail(
            claimed,
            attempt_state="SALIDA_INVALIDA",
            error_code="invalid_structured_output",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM returned invalid output") from error
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        reason = (
            str(error).splitlines()[0][:120] if isinstance(error, ValueError) else "unclassified"
        )
        validation_error = reason if reason in VALIDATION_CORRECTIONS else None
        logger.warning(
            "Generation request %s failed during LLM processing (%s; status=%s; reason=%s)",
            request_id,
            type(error).__name__,
            error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None,
            reason,
        )
        exhausted = await repository.fail(
            claimed,
            attempt_state="SALIDA_INVALIDA" if validation_error else "FALLIDO",
            error_code=validation_error or "llm_request_failed",
            max_attempts=settings.max_attempts,
        )
        if not exhausted:
            raise GenerationProcessingError("LLM request failed") from error
