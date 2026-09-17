import uuid
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

import gym_engine.api.routes as routes_module
from gym_engine.api.app import create_app
from gym_engine.api.routes import get_connection_dep
from gym_engine.config import Settings, get_settings
from tests.conftest import FakeConnection, FakeRepository


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "database_url": "postgresql://example.invalid/gym-test",
        "app_env": "test",
        "ai_service_api_key_test": "the-real-key",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def make_client(
    fake_repository: FakeRepository,
    fake_connection: FakeConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Settings], TestClient]:
    monkeypatch.setattr(routes_module, "repository", fake_repository)

    def _make(settings: Settings) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_connection_dep] = lambda: fake_connection
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)

    return _make


def _payload() -> dict:
    return {
        "idempotency_key": str(uuid.uuid4()),
        "gym_id": str(uuid.uuid4()),
        "student_id": str(uuid.uuid4()),
        "requested_by_user_id": str(uuid.uuid4()),
        "texto_libre": "quiero una rutina",
        "catalogo_prefiltrado": [
            {"id": str(uuid.uuid4()), "nombre": "Sentadilla", "patron_movimiento": "squat"}
        ],
        "contexto_minimizado": {"nivel_experiencia": "intermedio", "dias_semanales_disponibles": 3},
    }


def test_missing_api_key_is_rejected(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings())
    response = client.post("/v1/routine-generations", json=_payload())
    assert response.status_code == 401


def test_wrong_api_key_is_rejected(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings())
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


def test_correct_api_key_is_accepted(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings())
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "the-real-key"}
    )
    assert response.status_code == 202


def test_production_uses_production_key(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings(app_env="production", ai_service_api_key_production="prod-key"))
    # la clave de test ya no vale en produccion
    rejected = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "the-real-key"}
    )
    assert rejected.status_code == 401

    accepted = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "prod-key"}
    )
    assert accepted.status_code == 202


def test_unconfigured_key_fails_closed(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings(ai_service_api_key_test=None))
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "anything"}
    )
    assert response.status_code == 500


def test_get_endpoint_also_requires_api_key(make_client: Callable[[Settings], TestClient]) -> None:
    client = make_client(_settings())
    response = client.get(f"/v1/routine-generations/{uuid.uuid4()}")
    assert response.status_code == 401
