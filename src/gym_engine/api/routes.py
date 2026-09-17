from collections.abc import Iterator
from typing import Annotated, Any
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg.rows import DictRow

from gym_engine.api.auth import RequireApiKey
from gym_engine.api.schemas import (
    RoutineGenerationAccepted,
    RoutineGenerationCreate,
    RoutineGenerationStatus,
)
from gym_engine.config import Settings, get_settings
from gym_engine.persistence import repository
from gym_engine.persistence.db import get_connection

router = APIRouter(
    prefix="/v1/routine-generations",
    tags=["routine-generations"],
    dependencies=[RequireApiKey],
)

# codigos de error "conocidos" que un intento de LLM puede dejar en ai_generation_attempts.
# cualquier otro valor en error_code se interpreta como violaciones de negocio unidas con "; "
# y se devuelve desglosado en `violaciones` (ver _violations_from_error_code).
_STANDARD_ERROR_CODES = {
    "llm_timeout",
    "invalid_structured_output",
    "llm_request_failed",
    "unexpected_error",
}

_STATE_TO_STATUS = {
    "PENDIENTE": "pending",
    "PROCESANDO": "processing",
    "COMPLETADA": "completed",
    "NO_DISPONIBLE": "failed",
    "CANCELADA": "failed",
}


def get_connection_dep(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[psycopg.Connection[DictRow]]:
    conn = get_connection(settings)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


ConnectionDep = Annotated[psycopg.Connection[DictRow], Depends(get_connection_dep)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _build_minimized_context(payload: RoutineGenerationCreate) -> dict[str, Any]:
    """Empaqueta lo que el grafo necesita para reanudar. Las tablas de ai_integration no
    tienen columnas propias para esto -- minimized_context/preferences son jsonb libre."""
    return {
        "texto_libre": payload.texto_libre,
        "parametros": payload.parametros.model_dump(mode="json") if payload.parametros else None,
        "catalogo_prefiltrado": [e.model_dump(mode="json") for e in payload.catalogo_prefiltrado],
        "contexto_minimizado": payload.contexto_minimizado.model_dump(mode="json"),
    }


def _violations_from_error_code(error_code: str | None) -> list[str] | None:
    if error_code and error_code not in _STANDARD_ERROR_CODES:
        return error_code.split("; ")
    return None


@router.post("", response_model=RoutineGenerationAccepted, status_code=202)
def create_routine_generation(
    payload: RoutineGenerationCreate,
    response: Response,
    conn: ConnectionDep,
    settings: SettingsDep,
) -> RoutineGenerationAccepted:
    existing = repository.get_by_idempotency_key(conn, payload.idempotency_key)
    if existing is not None:
        response.status_code = 200
        return RoutineGenerationAccepted(
            request_id=existing["id"], status=_STATE_TO_STATUS[existing["state"]]
        )

    request = repository.create_request(
        conn,
        idempotency_key=payload.idempotency_key,
        minimized_context=_build_minimized_context(payload),
        preferences={},
        retention_days=settings.failed_result_retention_days,
    )
    return RoutineGenerationAccepted(
        request_id=request["id"], status=_STATE_TO_STATUS[request["state"]]
    )


@router.get("/{request_id}", response_model=RoutineGenerationStatus)
def get_routine_generation(request_id: UUID, conn: ConnectionDep) -> RoutineGenerationStatus:
    status = repository.get_status(conn, request_id)
    if status is None:
        raise HTTPException(status_code=404, detail="request_id no encontrado")

    return RoutineGenerationStatus(
        request_id=status.request_id,
        status=_STATE_TO_STATUS[status.state],
        estructura_candidata=status.structured_output,
        violaciones=_violations_from_error_code(status.error_code),
        error=status.error_code if status.state == "NO_DISPONIBLE" else None,
    )
