import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from gym_engine.api.app import create_app
from gym_engine.api.routes import get_session
from gym_engine.config import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "database_url": "sqlite://",
        "app_env": "test",
        "ai_service_api_key_test": "the-real-key",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def make_client(sqlite_session_factory: sessionmaker[Session]):
    def _make(settings: Settings) -> TestClient:
        app = create_app()

        def override_get_session() -> Iterator[Session]:
            db_session = sqlite_session_factory()
            try:
                yield db_session
                db_session.commit()
            finally:
                db_session.close()

        app.dependency_overrides[get_session] = override_get_session
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


def test_missing_api_key_is_rejected(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(_settings())
    response = client.post("/v1/routine-generations", json=_payload())
    assert response.status_code == 401


def test_wrong_api_key_is_rejected(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(_settings())
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


def test_correct_api_key_is_accepted(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(_settings())
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "the-real-key"}
    )
    assert response.status_code == 202


def test_production_uses_production_key(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(
        _settings(app_env="production", ai_service_api_key_production="prod-key")
    )
    # la clave de test ya no vale en produccion
    rejected = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "the-real-key"}
    )
    assert rejected.status_code == 401

    accepted = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "prod-key"}
    )
    assert accepted.status_code == 202


def test_unconfigured_key_fails_closed(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(_settings(ai_service_api_key_test=None))
    response = client.post(
        "/v1/routine-generations", json=_payload(), headers={"X-API-Key": "anything"}
    )
    assert response.status_code == 500


def test_get_endpoint_also_requires_api_key(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(_settings())
    response = client.get(f"/v1/routine-generations/{uuid.uuid4()}")
    assert response.status_code == 401
