from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from vercel.queue import QueueClient

from gym_engine.api import app as api_module
from gym_engine.config import Settings
from gym_engine.llm import OllamaClient
from gym_engine.persistence import GenerationRepository


def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql://example.invalid/gym-test",
        api_key="service-secret",
        queue_region="gru1",
        llm_api_url="https://llm.example.invalid",
        llm_model="qwen2.5:7b-instruct",
        llm_api_token="llm-secret",
        configuration_version="generative/generar-rutina@1",
        timeout_seconds=120,
        max_attempts=2,
    )


def test_health_does_not_require_dependencies() -> None:
    response = TestClient(api_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_service_authentication(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "get_settings", settings)
    response = TestClient(api_module.app).get("/ready")

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


async def _ping_ok(_self: object) -> None:
    return None


async def _ping_fails(_self: object) -> None:
    raise ConnectionError("unreachable")


def test_ready_reports_llm_down_when_database_is_up(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "ping", _ping_ok)
    monkeypatch.setattr(OllamaClient, "ping", _ping_fails)

    response = TestClient(api_module.app).get(
        "/ready", headers={"Authorization": "Bearer service-secret"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "dependency_unavailable",
            "database": "up",
            "llm": "down",
        }
    }


def test_ready_reports_database_down_when_llm_is_up(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "ping", _ping_fails)
    monkeypatch.setattr(OllamaClient, "ping", _ping_ok)

    response = TestClient(api_module.app).get(
        "/ready", headers={"Authorization": "Bearer service-secret"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "dependency_unavailable",
            "database": "down",
            "llm": "up",
        }
    }


def test_dispatches_only_an_existing_request(monkeypatch: MonkeyPatch) -> None:
    request_id = "83271cf7-9264-47b5-b85f-d09f05c99326"

    async def request_exists(
        _repository: GenerationRepository, _request_id: object
    ) -> bool:
        return True

    async def send(_queue: QueueClient, *_args: object, **_kwargs: object) -> str:
        return "msg_123"

    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "request_exists", request_exists)
    monkeypatch.setattr(QueueClient, "send", send)

    response = TestClient(api_module.app).post(
        f"/v1/generation-requests/{request_id}/dispatch",
        headers={"Authorization": "Bearer service-secret"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "request_id": request_id,
        "status": "queued",
        "message_id": "msg_123",
    }
