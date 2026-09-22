import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import gym_engine.api.routes as routes_module
from gym_engine.api.app import create_app
from gym_engine.api.auth import verify_api_key
from gym_engine.api.routes import get_connection_dep, get_queue_client
from gym_engine.config import Settings, get_settings
from tests.conftest import FakeConnection, FakeQueueClient, FakeRepository


def _test_settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql://example.invalid/gym-test",
        ai_service_api_key="service-secret",
    )


@pytest.fixture
def client(
    fake_repository: FakeRepository,
    fake_connection: FakeConnection,
    fake_queue_client: FakeQueueClient,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(routes_module, "repository", fake_repository)
    app = create_app()
    app.dependency_overrides[get_connection_dep] = lambda: fake_connection
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_settings] = _test_settings
    app.dependency_overrides[get_queue_client] = lambda: fake_queue_client
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


def test_create_returns_202_and_pending(
    client: TestClient, fake_queue_client: FakeQueueClient
) -> None:
    response = client.post("/v1/routine-generations", json=_payload())
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert uuid.UUID(body["request_id"])
    assert len(fake_queue_client.sent) == 1
    assert fake_queue_client.sent[0]["payload"] == {"request_id": body["request_id"]}
    assert fake_queue_client.sent[0]["idempotency_key"] == body["request_id"]


def test_create_is_idempotent(client: TestClient, fake_queue_client: FakeQueueClient) -> None:
    first = client.post("/v1/routine-generations", json=_payload("same-key")).json()
    second_response = client.post("/v1/routine-generations", json=_payload("same-key"))
    assert second_response.status_code == 200
    assert second_response.json()["request_id"] == first["request_id"]
    # La rama idempotente no debe publicar un segundo mensaje.
    assert len(fake_queue_client.sent) == 1


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


def test_get_completed_request_returns_structured_output(
    client: TestClient, fake_repository: FakeRepository
) -> None:
    created = client.post("/v1/routine-generations", json=_payload("idem-completed")).json()
    request_id = uuid.UUID(created["request_id"])
    fake_repository.set_state(
        request_id, state="COMPLETADA", structured_output={"dias": []}
    )

    response = client.get(f"/v1/routine-generations/{request_id}")
    body = response.json()
    assert body["status"] == "completed"
    assert body["estructura_candidata"] == {"dias": []}


def test_get_failed_request_returns_error(
    client: TestClient, fake_repository: FakeRepository
) -> None:
    created = client.post("/v1/routine-generations", json=_payload("idem-failed")).json()
    request_id = uuid.UUID(created["request_id"])
    fake_repository.set_state(request_id, state="NO_DISPONIBLE", error_code="llm_timeout")

    response = client.get(f"/v1/routine-generations/{request_id}")
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == "llm_timeout"


def test_get_failed_request_with_business_violations(
    client: TestClient, fake_repository: FakeRepository
) -> None:
    created = client.post("/v1/routine-generations", json=_payload("idem-invalid")).json()
    request_id = uuid.UUID(created["request_id"])
    fake_repository.set_state(
        request_id,
        state="NO_DISPONIBLE",
        error_code="objetivo vacio; frecuencia_semanal fuera de RN-38 (1-7)",
    )

    response = client.get(f"/v1/routine-generations/{request_id}")
    body = response.json()
    assert body["violaciones"] == ["objetivo vacio", "frecuencia_semanal fuera de RN-38 (1-7)"]


def test_health_does_not_require_dependencies() -> None:
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_service_authentication() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = _test_settings

    response = TestClient(app).get("/ready")

    assert response.status_code == 401


def test_ready_accepts_bearer_authorization() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = _test_settings

    response = TestClient(app).get("/ready", headers={"Authorization": "Bearer service-secret"})

    # Sin Postgres real en este entorno de tests (ver docstring de conftest.py): lo que
    # importa acá es que el header Bearer pasó la autenticación (no 401), no que la DB
    # responda 200.
    assert response.status_code == 503
