from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from logging import getLogger
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, status
from vercel.queue import (
    DuplicateIdempotencyKeyError,
    QueueClient,
    QueueError,
)

from ..config import ConfigurationError, Settings
from ..contracts import DispatchResponse, GenerationMessage
from ..llm import OllamaClient
from ..persistence import GenerationRepository
from ..service import GenerationProcessingError, process_generation

logger = getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


async def process_generation_locally(
    request_id: UUID,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        for delivery_attempt in range(settings.max_attempts):
            try:
                await process_generation(request_id, settings)
                return
            except GenerationProcessingError:
                if delivery_attempt + 1 >= settings.max_attempts:
                    raise
                logger.warning(
                    "Retrying local generation request %s (delivery %d)",
                    request_id,
                    delivery_attempt + 2,
                )


async def recover_local_retryable_generations(
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> None:
    request_ids = await GenerationRepository(
        settings.database_url
    ).retryable_request_ids()
    for request_id in request_ids:
        try:
            await process_generation_locally(request_id, settings, semaphore)
        except Exception as error:
            logger.warning(
                "Could not recover local generation %s (%s)",
                request_id,
                type(error).__name__,
            )


async def run_local_generation_worker(
    settings: Settings,
    semaphore: asyncio.Semaphore,
    wakeup: asyncio.Event,
    poll_seconds: float = 2,
) -> None:
    while True:
        wakeup.clear()
        try:
            await recover_local_retryable_generations(settings, semaphore)
        except Exception as error:
            logger.warning("Local generation queue check failed (%s)", type(error).__name__)
        with suppress(TimeoutError):
            await asyncio.wait_for(wakeup.wait(), timeout=poll_seconds)


async def require_service_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    try:
        expected = get_settings().api_key
    except ConfigurationError as error:
        raise HTTPException(status_code=503, detail="service_not_configured") from error

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app() -> FastAPI:
    local_worker_semaphore = asyncio.Semaphore(1)
    local_worker_wakeup = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        recovery_task: asyncio.Task[None] | None = None
        local_mode = os.getenv("GENERATION_QUEUE_MODE", "vercel").strip().lower()
        if local_mode == "local":
            try:
                settings = get_settings()
            except ConfigurationError as error:
                logger.warning(
                    "Local generation recovery is not configured (%s)",
                    type(error).__name__,
                )
            else:
                recovery_task = asyncio.create_task(
                    run_local_generation_worker(
                        settings,
                        local_worker_semaphore,
                        local_worker_wakeup,
                    ),
                    name="local-generation-worker",
                )
        try:
            yield
        finally:
            if recovery_task is not None:
                recovery_task.cancel()
                with suppress(asyncio.CancelledError):
                    await recovery_task

    application = FastAPI(
        title="Proyecto Gimnasio - Servicio IA",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @application.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get(
        "/ready", tags=["operations"], dependencies=[Depends(require_service_key)]
    )
    async def ready() -> dict[str, str]:
        try:
            settings = get_settings()
            await GenerationRepository(settings.database_url).ping()
        except Exception as error:
            loop_name = type(asyncio.get_running_loop()).__name__
            logger.warning(
                "Readiness check failed for database (%s; event_loop=%s)",
                type(error).__name__,
                loop_name,
            )
            raise HTTPException(status_code=503, detail="dependency_unavailable") from error

        try:
            await OllamaClient(
                settings.llm_api_url,
                settings.llm_model,
                settings.llm_api_token,
                settings.timeout_seconds,
            ).ping()
        except Exception as error:
            logger.warning("Readiness check failed for LLM (%s)", type(error).__name__)
            raise HTTPException(status_code=503, detail="dependency_unavailable") from error
        return {"status": "ready", "database": "up", "llm": "up"}

    @application.post(
        "/v1/generation-requests/{request_id}/dispatch",
        response_model=DispatchResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["generation"],
        dependencies=[Depends(require_service_key)],
    )
    async def dispatch_generation(
        request_id: UUID,
    ) -> DispatchResponse:
        settings = get_settings()
        repository = GenerationRepository(settings.database_url)
        if not await repository.request_exists(request_id):
            raise HTTPException(status_code=404, detail="generation_request_not_found")

        if settings.generation_queue_mode == "local":
            logger.info(
                "Dispatching generation %s to the local development worker",
                request_id,
            )
            local_worker_wakeup.set()
            return DispatchResponse(request_id=request_id)

        queue = QueueClient(region=settings.queue_region)
        try:
            message_id = await queue.send(
                "gym-routine-generations",
                GenerationMessage(request_id=request_id),
                idempotency_key=str(request_id),
            )
        except DuplicateIdempotencyKeyError:
            return DispatchResponse(request_id=request_id)
        except QueueError as error:
            logger.warning("Queue dispatch failed (%s)", type(error).__name__)
            raise HTTPException(
                status_code=503, detail="generation_queue_unavailable"
            ) from error
        return DispatchResponse(request_id=request_id, message_id=message_id)

    return application


app = create_app()
