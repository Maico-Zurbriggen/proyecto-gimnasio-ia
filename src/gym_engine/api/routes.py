from collections.abc import Iterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from gym_engine.api.auth import RequireApiKey
from gym_engine.api.schemas import (
    RoutineGenerationAccepted,
    RoutineGenerationCreate,
    RoutineGenerationStatus,
)
from gym_engine.persistence import repository
from gym_engine.persistence.db import get_session_factory
from gym_engine.persistence.models import AiGenerationResult, AiResultValidation

router = APIRouter(
    prefix="/v1/routine-generations",
    tags=["routine-generations"],
    dependencies=[RequireApiKey],
)


def get_session() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


@router.post("", response_model=RoutineGenerationAccepted, status_code=202)
def create_routine_generation(
    payload: RoutineGenerationCreate, response: Response, session: SessionDep
) -> RoutineGenerationAccepted:
    existing = repository.get_by_idempotency_key(session, payload.idempotency_key)
    if existing is not None:
        response.status_code = 200
        return RoutineGenerationAccepted(request_id=existing.id, status=existing.status)

    request = repository.create_request(
        session,
        idempotency_key=payload.idempotency_key,
        gym_id=payload.gym_id,
        student_id=payload.student_id,
        requested_by_user_id=payload.requested_by_user_id,
        catalogo_prefiltrado=[e.model_dump(mode="json") for e in payload.catalogo_prefiltrado],
        contexto_minimizado=payload.contexto_minimizado.model_dump(mode="json"),
        texto_libre=payload.texto_libre,
        parametros=payload.parametros.model_dump(mode="json") if payload.parametros else None,
    )
    return RoutineGenerationAccepted(request_id=request.id, status=request.status)


@router.get("/{request_id}", response_model=RoutineGenerationStatus)
def get_routine_generation(request_id: UUID, session: SessionDep) -> RoutineGenerationStatus:
    request = repository.get_request(session, request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="request_id no encontrado")

    result = session.execute(
        select(AiGenerationResult)
        .where(AiGenerationResult.request_id == request_id)
        .order_by(AiGenerationResult.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    validation = session.execute(
        select(AiResultValidation)
        .where(AiResultValidation.request_id == request_id)
        .order_by(AiResultValidation.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    return RoutineGenerationStatus(
        request_id=request.id,
        status=request.status,
        estructura_candidata=result.estructura_candidata if result else None,
        violaciones=validation.violations if validation else None,
        error=request.error,
    )
