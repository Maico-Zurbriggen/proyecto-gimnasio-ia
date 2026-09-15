from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, status
from vercel.queue import QueueClient

from ..config import ConfigurationError, Settings
from ..contracts import DispatchResponse, GenerationMessage
from ..llm import OllamaClient
from ..persistence import GenerationRepository


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


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
    application = FastAPI(
        title="Proyecto Gimnasio - Servicio IA",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
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
            await OllamaClient(
                settings.llm_api_url,
                settings.llm_model,
                settings.llm_api_token,
                settings.timeout_seconds,
            ).ping()
        except Exception as error:
            raise HTTPException(status_code=503, detail="dependency_unavailable") from error
        return {"status": "ready", "database": "up", "llm": "up"}

    @application.post(
        "/v1/generation-requests/{request_id}/dispatch",
        response_model=DispatchResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["generation"],
        dependencies=[Depends(require_service_key)],
    )
    async def dispatch_generation(request_id: UUID) -> DispatchResponse:
        settings = get_settings()
        repository = GenerationRepository(settings.database_url)
        if not await repository.request_exists(request_id):
            raise HTTPException(status_code=404, detail="generation_request_not_found")

        queue = QueueClient(region=settings.queue_region)
        message_id = await queue.send(
            "gym-routine-generations",
            GenerationMessage(request_id=request_id),
            idempotency_key=str(request_id),
        )
        return DispatchResponse(request_id=request_id, message_id=message_id)

    return application


app = create_app()
