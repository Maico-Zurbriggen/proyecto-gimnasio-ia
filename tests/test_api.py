import asyncio
from dataclasses import replace
from threading import Event
from uuid import UUID

from fastapi.testclient import TestClient
from pytest import MonkeyPatch, raises
from vercel.queue import (
    DuplicateIdempotencyKeyError,
    QueueClient,
    TokenResolutionError,
)

from gym_engine.api import app as api_module
from gym_engine.config import ConfigurationError, Settings
from gym_engine.persistence import GenerationRepository
from gym_engine.service import GenerationProcessingError


def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql://example.invalid/gym-test",
        api_key="service-secret",
        generation_queue_mode="vercel",
        queue_region="gru1",
        llm_api_url="https://llm.example.invalid",
        llm_model="qwen3.5:9b-instruct",
        llm_api_token="llm-secret",
        configuration_version="generative/generar-rutina@1",
        timeout_seconds=120,
        max_attempts=2,
    )


def test_health_does_not_require_dependencies() -> None:
    response = TestClient(api_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_queue_mode_is_rejected_outside_local_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("GENERATION_QUEUE_MODE", "local")

    with raises(ConfigurationError, match="only allowed with APP_ENV=local"):
        Settings.from_env()


def test_local_worker_retries_a_recorded_generation_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    request_id = UUID("83271cf7-9264-47b5-b85f-d09f05c99326")
    calls = 0

    async def process(_request_id: UUID, _settings: Settings) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GenerationProcessingError("retryable failure")

    monkeypatch.setattr(api_module, "process_generation", process)

    asyncio.run(
        api_module.process_generation_locally(
            request_id,
            settings(),
            asyncio.Semaphore(1),
        )
    )

    assert calls == 2


def test_local_environment_rejects_a_remote_database(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("GENERATION_QUEUE_MODE", "local")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/gym-test")

    with raises(ConfigurationError, match="requires a local PostgreSQL"):
        Settings.from_env()


def test_local_startup_recovers_pending_failed_attempts(
    monkeypatch: MonkeyPatch,
) -> None:
    request_id = UUID("83271cf7-9264-47b5-b85f-d09f05c99326")
    processed: list[UUID] = []

    async def list_retryable(
        _repository: GenerationRepository,
    ) -> list[UUID]:
        return [request_id]

    async def process(actual_request_id: UUID, _settings: Settings) -> None:
        processed.append(actual_request_id)

    monkeypatch.setattr(
        GenerationRepository,
        "retryable_request_ids",
        list_retryable,
    )
    monkeypatch.setattr(api_module, "process_generation", process)

    asyncio.run(
        api_module.recover_local_retryable_generations(
            settings(),
            asyncio.Semaphore(1),
        )
    )

    assert processed == [request_id]


def test_ready_requires_service_authentication(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "get_settings", settings)
    response = TestClient(api_module.app).get("/ready")

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_ready_checks_database_and_llm_with_bearer_auth(
    monkeypatch: MonkeyPatch,
) -> None:
    async def ping(_dependency: object) -> None:
        return None

    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "ping", ping)
    monkeypatch.setattr(api_module.OllamaClient, "ping", ping)

    response = TestClient(api_module.app).get(
        "/ready",
        headers={"Authorization": "Bearer service-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "up", "llm": "up"}


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


def test_local_dispatch_wakes_persistent_worker_without_vercel_queue(
    monkeypatch: MonkeyPatch,
) -> None:
    request_id = "83271cf7-9264-47b5-b85f-d09f05c99326"
    local_settings = replace(settings(), app_env="local", generation_queue_mode="local")
    processed: list[tuple[object, Settings]] = []
    finished = Event()
    available = False

    async def request_exists(
        _repository: GenerationRepository, _request_id: object
    ) -> bool:
        nonlocal available
        available = True
        return True

    async def list_retryable(_repository: GenerationRepository) -> list[UUID]:
        return [UUID(request_id)] if available else []

    async def process_generation(
        actual_request_id: object, actual_settings: Settings
    ) -> None:
        nonlocal available
        processed.append((actual_request_id, actual_settings))
        available = False
        finished.set()

    monkeypatch.setenv("GENERATION_QUEUE_MODE", "local")
    monkeypatch.setattr(api_module, "get_settings", lambda: local_settings)
    monkeypatch.setattr(GenerationRepository, "request_exists", request_exists)
    monkeypatch.setattr(GenerationRepository, "retryable_request_ids", list_retryable)
    monkeypatch.setattr(api_module, "process_generation", process_generation)

    with TestClient(api_module.create_app()) as client:
        response = client.post(
            f"/v1/generation-requests/{request_id}/dispatch",
            headers={"Authorization": "Bearer service-secret"},
        )
        assert finished.wait(timeout=1)

    assert response.status_code == 202
    assert response.json() == {
        "request_id": request_id,
        "status": "queued",
        "message_id": None,
    }
    assert len(processed) == 1
    assert str(processed[0][0]) == request_id
    assert processed[0][1] == local_settings


def test_local_worker_keeps_polling_after_a_temporary_database_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    async def run() -> None:
        finished = asyncio.Event()

        async def scan(_settings: Settings, _semaphore: asyncio.Semaphore) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("database temporarily unavailable")
            finished.set()

        monkeypatch.setattr(api_module, "recover_local_retryable_generations", scan)
        worker = asyncio.create_task(
            api_module.run_local_generation_worker(
                settings(), asyncio.Semaphore(1), asyncio.Event(), poll_seconds=0.01
            )
        )
        try:
            await asyncio.wait_for(finished.wait(), timeout=1)
        finally:
            worker.cancel()
            with raises(asyncio.CancelledError):
                await worker

    asyncio.run(run())
    assert calls >= 2


def test_dispatch_does_not_queue_a_missing_request(monkeypatch: MonkeyPatch) -> None:
    async def request_does_not_exist(
        _repository: GenerationRepository, _request_id: object
    ) -> bool:
        return False

    async def send(_queue: QueueClient, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("Queue must not receive an unknown request")

    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(
        GenerationRepository, "request_exists", request_does_not_exist
    )
    monkeypatch.setattr(QueueClient, "send", send)

    response = TestClient(api_module.app).post(
        "/v1/generation-requests/83271cf7-9264-47b5-b85f-d09f05c99326/dispatch",
        headers={"Authorization": "Bearer service-secret"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "generation_request_not_found"}


def test_dispatch_returns_retryable_unavailable_without_queue_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    async def request_exists(
        _repository: GenerationRepository, _request_id: object
    ) -> bool:
        return True

    async def send(_queue: QueueClient, *_args: object, **_kwargs: object) -> str:
        raise TokenResolutionError("missing queue token")

    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "request_exists", request_exists)
    monkeypatch.setattr(QueueClient, "send", send)

    response = TestClient(api_module.app).post(
        "/v1/generation-requests/83271cf7-9264-47b5-b85f-d09f05c99326/dispatch",
        headers={"Authorization": "Bearer service-secret"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "generation_queue_unavailable"}


def test_dispatch_treats_an_idempotent_queue_duplicate_as_accepted(
    monkeypatch: MonkeyPatch,
) -> None:
    async def request_exists(
        _repository: GenerationRepository, _request_id: object
    ) -> bool:
        return True

    async def send(_queue: QueueClient, *_args: object, **_kwargs: object) -> str:
        raise DuplicateIdempotencyKeyError("duplicate")

    monkeypatch.setattr(api_module, "get_settings", settings)
    monkeypatch.setattr(GenerationRepository, "request_exists", request_exists)
    monkeypatch.setattr(QueueClient, "send", send)

    response = TestClient(api_module.app).post(
        "/v1/generation-requests/83271cf7-9264-47b5-b85f-d09f05c99326/dispatch",
        headers={"Authorization": "Bearer service-secret"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "request_id": "83271cf7-9264-47b5-b85f-d09f05c99326",
        "status": "queued",
        "message_id": None,
    }
