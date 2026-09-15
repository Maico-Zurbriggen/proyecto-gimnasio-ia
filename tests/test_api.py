import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from gym_engine.api.app import create_app
from gym_engine.api.auth import verify_api_key
from gym_engine.api.routes import get_session


@pytest.fixture
def client(sqlite_session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    app = create_app()

    def override_get_session() -> Iterator[Session]:
        db_session = sqlite_session_factory()
        try:
            yield db_session
            db_session.commit()
        finally:
            db_session.close()

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def _payload(idempotency_key: str = "idem-1") -> dict:
    return {
        "idempotency_key": idempotency_key,
        "gym_id": str(uuid.uuid4()),
        "student_id": str(uuid.uuid4()),
        "requested_by_user_id": str(uuid.uuid4()),
        "texto_libre": "quiero una rutina de fuerza",
        "catalogo_prefiltrado": [
            {"id": str(uuid.uuid4()), "nombre": "Sentadilla", "patron_movimiento": "squat"}
        ],
        "contexto_minimizado": {"nivel_experiencia": "intermedio", "dias_semanales_disponibles": 4},
    }


def test_create_returns_202_and_pending(client: TestClient) -> None:
    response = client.post("/v1/routine-generations", json=_payload())
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert uuid.UUID(body["request_id"])


def test_create_is_idempotent(client: TestClient) -> None:
    first = client.post("/v1/routine-generations", json=_payload("same-key")).json()
    second_response = client.post("/v1/routine-generations", json=_payload("same-key"))
    assert second_response.status_code == 200
    assert second_response.json()["request_id"] == first["request_id"]


def test_create_rejects_empty_catalogo(client: TestClient) -> None:
    payload = _payload()
    payload["catalogo_prefiltrado"] = []
    response = client.post("/v1/routine-generations", json=payload)
    assert response.status_code == 422


def test_get_unknown_request_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/routine-generations/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_pending_request(client: TestClient) -> None:
    created = client.post("/v1/routine-generations", json=_payload("idem-status")).json()
    response = client.get(f"/v1/routine-generations/{created['request_id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
