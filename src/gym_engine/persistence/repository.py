import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from gym_engine.persistence.models import (
    AiGenerationAttempt,
    AiGenerationRequest,
    AiGenerationResult,
    AiResultValidation,
)


def get_by_idempotency_key(session: Session, idempotency_key: str) -> AiGenerationRequest | None:
    stmt = select(AiGenerationRequest).where(AiGenerationRequest.idempotency_key == idempotency_key)
    return session.execute(stmt).scalar_one_or_none()


def create_request(
    session: Session,
    *,
    idempotency_key: str,
    gym_id: uuid.UUID,
    student_id: uuid.UUID,
    requested_by_user_id: uuid.UUID,
    catalogo_prefiltrado: list[dict[str, Any]],
    contexto_minimizado: dict[str, Any],
    texto_libre: str | None = None,
    parametros: dict[str, Any] | None = None,
) -> AiGenerationRequest:
    request = AiGenerationRequest(
        idempotency_key=idempotency_key,
        gym_id=gym_id,
        student_id=student_id,
        requested_by_user_id=requested_by_user_id,
        catalogo_prefiltrado=catalogo_prefiltrado,
        contexto_minimizado=contexto_minimizado,
        texto_libre=texto_libre,
        parametros=parametros,
        status="pending",
    )
    session.add(request)
    session.flush()
    return request


def get_request(session: Session, request_id: uuid.UUID) -> AiGenerationRequest | None:
    return session.get(AiGenerationRequest, request_id)


def claim_next_pending(session: Session) -> AiGenerationRequest | None:
    """Reclamo atómico de una solicitud pendiente para este worker (evita doble procesamiento).

    ``FOR UPDATE SKIP LOCKED`` sólo lo soporta Postgres; en otros dialectos (sqlite en tests)
    se omite y se confía en la condición ``status == "pending"`` del UPDATE siguiente.
    """
    query = (
        select(AiGenerationRequest.id)
        .where(AiGenerationRequest.status == "pending")
        .order_by(AiGenerationRequest.created_at)
        .limit(1)
    )
    if session.get_bind().dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    candidate = session.execute(query).scalar_one_or_none()
    if candidate is None:
        return None

    session.execute(
        update(AiGenerationRequest)
        .where(AiGenerationRequest.id == candidate, AiGenerationRequest.status == "pending")
        .values(status="processing")
    )
    session.flush()
    return session.get(AiGenerationRequest, candidate)


def record_attempt(
    session: Session, *, request_id: uuid.UUID, node: str, attempt_number: int
) -> AiGenerationAttempt:
    attempt = AiGenerationAttempt(request_id=request_id, node=node, attempt_number=attempt_number)
    session.add(attempt)
    session.flush()
    return attempt


def finish_attempt(
    session: Session, attempt: AiGenerationAttempt, *, error: str | None = None
) -> None:
    attempt.finished_at = datetime.now(UTC)
    attempt.error = error
    session.flush()


def save_result(
    session: Session,
    *,
    request_id: uuid.UUID,
    estructura_candidata: dict[str, Any],
    version_modelo: str,
    version_prompt: str,
) -> AiGenerationResult:
    result = AiGenerationResult(
        request_id=request_id,
        estructura_candidata=estructura_candidata,
        version_modelo=version_modelo,
        version_prompt=version_prompt,
    )
    session.add(result)
    session.execute(
        update(AiGenerationRequest)
        .where(AiGenerationRequest.id == request_id)
        .values(status="completed")
    )
    session.flush()
    return result


def save_validation(
    session: Session, *, request_id: uuid.UUID, valid: bool, violations: list[str]
) -> AiResultValidation:
    validation = AiResultValidation(request_id=request_id, valid=valid, violations=violations)
    session.add(validation)
    session.flush()
    return validation


def mark_failed(session: Session, *, request_id: uuid.UUID, error: str) -> None:
    session.execute(
        update(AiGenerationRequest)
        .where(AiGenerationRequest.id == request_id)
        .values(status="failed", error=error)
    )
    session.flush()
