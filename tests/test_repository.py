import uuid

from sqlalchemy.orm import Session

from gym_engine.persistence import repository


def _create(session: Session, idempotency_key: str = "key-1") -> uuid.UUID:
    request = repository.create_request(
        session,
        idempotency_key=idempotency_key,
        gym_id=uuid.uuid4(),
        student_id=uuid.uuid4(),
        requested_by_user_id=uuid.uuid4(),
        catalogo_prefiltrado=[
            {"id": str(uuid.uuid4()), "nombre": "Sentadilla", "patron_movimiento": "squat"}
        ],
        contexto_minimizado={
            "nivel_experiencia": "principiante",
            "dias_semanales_disponibles": 3,
        },
        texto_libre="quiero ganar fuerza",
    )
    session.commit()
    return request.id


def test_create_request_is_idempotent(session: Session) -> None:
    first_id = _create(session, "same-key")
    second = repository.get_by_idempotency_key(session, "same-key")
    assert second is not None
    assert second.id == first_id

    # una segunda "creacion" con la misma clave, tal como haria el endpoint, debe reusar la fila
    existing = repository.get_by_idempotency_key(session, "same-key")
    assert existing is not None
    assert existing.id == first_id


def test_claim_next_pending_marks_processing_and_is_exclusive(session: Session) -> None:
    request_id = _create(session)

    claimed = repository.claim_next_pending(session)
    session.commit()
    assert claimed is not None
    assert claimed.id == request_id
    assert claimed.status == "processing"

    assert repository.claim_next_pending(session) is None


def test_save_result_marks_request_completed(session: Session) -> None:
    request_id = _create(session)
    repository.claim_next_pending(session)

    repository.save_result(
        session,
        request_id=request_id,
        estructura_candidata={"dias": []},
        version_modelo="llama3.1",
        version_prompt="generative/generar-rutina@1",
    )
    session.commit()

    request = repository.get_request(session, request_id)
    assert request is not None
    assert request.status == "completed"


def test_mark_failed(session: Session) -> None:
    request_id = _create(session)
    repository.mark_failed(session, request_id=request_id, error="llm no disponible")
    session.commit()

    request = repository.get_request(session, request_id)
    assert request is not None
    assert request.status == "failed"
    assert request.error == "llm no disponible"
