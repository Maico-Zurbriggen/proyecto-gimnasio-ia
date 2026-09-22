"""Testea el grafo cableado end-to-end (nodos + persistencia) y el consumer de Vercel Queues
mockeando la capa de persistencia -- no hay Postgres real disponible en este entorno para
probar claim_by_id/claim_next_pending/complete/fail contra las tablas reales de
ai_integration."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from vercel.queue import RetryAfter

import gym_engine.orchestration.nodes as nodes_module
import gym_engine.worker.queue_consumer as consumer_module
from gym_engine.config import Settings
from gym_engine.llm.schemas import (
    DiaRutinaCandidato,
    EjercicioRutinaCandidato,
    ParametrosRutina,
    RutinaEstructurada,
    SeriePrescriptaCandidata,
)
from gym_engine.orchestration.graph import build_graph
from gym_engine.orchestration.nodes import NodeDeps
from gym_engine.persistence.models import ClaimedGeneration
from gym_engine.worker.queue_consumer import _initial_state, handle_routine_generation


class FakeGenerationClient:
    def __init__(self, ejercicio_id: uuid.UUID) -> None:
        self._ejercicio_id = ejercicio_id

    def generate_structured(self, prompt: str, schema: type) -> object:
        if schema is RutinaEstructurada:
            return RutinaEstructurada(
                dias=[
                    DiaRutinaCandidato(
                        orden=1,
                        nombre="Dia 1",
                        ejercicios=[
                            EjercicioRutinaCandidato(
                                ejercicio_id=self._ejercicio_id,
                                orden=1,
                                series=[
                                    SeriePrescriptaCandidata(
                                        orden=1,
                                        repeticiones_min=8,
                                        repeticiones_max=12,
                                        descanso_segundos=60,
                                    )
                                ],
                            )
                        ],
                    )
                ]
            )
        raise AssertionError(f"schema inesperado: {schema}")


class FakeRepositoryCalls:
    """Reemplaza gym_engine.persistence.repository en tests del grafo y del consumer.
    claim_by_id devuelve un resultado fijado de antemano (ver los tests de handler)."""

    def __init__(self, claim_result: ClaimedGeneration | None = None) -> None:
        self.completed: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self._claim_result = claim_result

    def claim_by_id(
        self, conn: object, request_id: uuid.UUID, **kwargs: object
    ) -> ClaimedGeneration | None:
        return self._claim_result

    def complete(self, conn: object, claimed: ClaimedGeneration, **kwargs: object) -> None:
        self.completed.append({"claimed": claimed, **kwargs})

    def fail(self, conn: object, claimed: ClaimedGeneration, **kwargs: object) -> bool:
        self.failed.append({"claimed": claimed, **kwargs})
        return kwargs.get("max_attempts") == 1


class FakeConn:
    def __init__(self) -> None:
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "database_url": "postgresql://example.invalid/gym-test",
        "ollama_model": "test-model",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _claimed(ejercicio_id: uuid.UUID) -> ClaimedGeneration:
    return ClaimedGeneration(
        request_id=uuid.uuid4(),
        attempt_id=uuid.uuid4(),
        attempt_number=1,
        minimized_context={
            "texto_libre": None,
            "parametros": ParametrosRutina(
                objetivo="fuerza", frecuencia_semanal=3, duracion_minutos=50
            ).model_dump(mode="json"),
            "catalogo_prefiltrado": [
                {"id": str(ejercicio_id), "nombre": "Sentadilla", "patron_movimiento": "squat"}
            ],
            "contexto_minimizado": {
                "nivel_experiencia": "intermedio",
                "dias_semanales_disponibles": 3,
            },
        },
        preferences={},
        retention_until=datetime.now(UTC),
    )


def _message(request_id: uuid.UUID) -> Any:
    """Stand-in de vercel.queue.Message: el handler solo lee .payload."""
    return SimpleNamespace(payload={"request_id": str(request_id)})


def test_process_completes_and_calls_repository_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ejercicio_id = uuid.uuid4()
    claimed = _claimed(ejercicio_id)
    calls = FakeRepositoryCalls()
    monkeypatch.setattr(nodes_module, "repository", calls)  # type: ignore[attr-defined]

    deps = NodeDeps(
        client=FakeGenerationClient(ejercicio_id),  # type: ignore[arg-type]
        conn=None,  # type: ignore[arg-type]
        settings=_settings(),
        claimed=claimed,
    )
    graph = build_graph(deps)
    graph.invoke(_initial_state(claimed))

    assert len(calls.completed) == 1
    assert calls.failed == []
    output = calls.completed[0]["output"]
    assert output["tipo_rutina"] == "FUERZA"
    assert output["frecuencia_semanal"] == 3
    assert output["dias"][0]["ejercicios"][0]["ejercicio_id"] == str(ejercicio_id)
    assert output["dias"][0]["ejercicios"][0]["series"][0]["carga_sugerida"] == 0


def test_process_fails_when_llm_returns_exercise_outside_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogo_id = uuid.uuid4()
    ajeno_id = uuid.uuid4()
    claimed = _claimed(catalogo_id)
    calls = FakeRepositoryCalls()
    monkeypatch.setattr(nodes_module, "repository", calls)  # type: ignore[attr-defined]

    deps = NodeDeps(
        client=FakeGenerationClient(ajeno_id),  # type: ignore[arg-type]
        conn=None,  # type: ignore[arg-type]
        settings=_settings(),
        claimed=claimed,
    )
    graph = build_graph(deps)
    graph.invoke(_initial_state(claimed))

    assert calls.completed == []
    assert len(calls.failed) == 1
    assert calls.failed[0]["attempt_state"] == "SALIDA_INVALIDA"
    assert "no pertenece al catalogo" in calls.failed[0]["error_code"]


def test_handler_acks_without_work_when_claim_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = FakeRepositoryCalls(claim_result=None)
    fake_conn = FakeConn()
    monkeypatch.setattr(consumer_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "get_connection", lambda settings: fake_conn)
    monkeypatch.setattr(consumer_module, "get_settings", _settings)

    handle_routine_generation(_message(uuid.uuid4()))

    assert fake_conn.committed == 1
    assert fake_conn.closed is True
    assert calls.completed == []
    assert calls.failed == []


def test_handler_raises_retry_after_on_non_exhausted_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogo_id = uuid.uuid4()
    ajeno_id = uuid.uuid4()
    claimed = _claimed(catalogo_id)
    calls = FakeRepositoryCalls(claim_result=claimed)
    fake_conn = FakeConn()
    settings = _settings()
    monkeypatch.setattr(nodes_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "get_connection", lambda s: fake_conn)
    monkeypatch.setattr(consumer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        consumer_module, "build_llm_client", lambda s: FakeGenerationClient(ajeno_id)
    )

    with pytest.raises(RetryAfter):
        handle_routine_generation(_message(claimed.request_id))

    assert calls.completed == []
    assert len(calls.failed) == 1
    assert fake_conn.closed is True


def test_handler_does_not_retry_when_attempts_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogo_id = uuid.uuid4()
    ajeno_id = uuid.uuid4()
    claimed = _claimed(catalogo_id)
    calls = FakeRepositoryCalls(claim_result=claimed)
    fake_conn = FakeConn()
    settings = _settings(generation_max_retries=0)
    monkeypatch.setattr(nodes_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "get_connection", lambda s: fake_conn)
    monkeypatch.setattr(consumer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        consumer_module, "build_llm_client", lambda s: FakeGenerationClient(ajeno_id)
    )

    handle_routine_generation(_message(claimed.request_id))

    assert len(calls.failed) == 1
    assert fake_conn.closed is True


def test_handler_completes_successfully_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ejercicio_id = uuid.uuid4()
    claimed = _claimed(ejercicio_id)
    calls = FakeRepositoryCalls(claim_result=claimed)
    fake_conn = FakeConn()
    settings = _settings()
    monkeypatch.setattr(nodes_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "repository", calls)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer_module, "get_connection", lambda s: fake_conn)
    monkeypatch.setattr(consumer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        consumer_module, "build_llm_client", lambda s: FakeGenerationClient(ejercicio_id)
    )

    handle_routine_generation(_message(claimed.request_id))

    assert len(calls.completed) == 1
    assert calls.failed == []
    assert fake_conn.closed is True
