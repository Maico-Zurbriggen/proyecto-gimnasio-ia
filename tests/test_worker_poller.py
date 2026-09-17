"""Testea el grafo cableado end-to-end (nodos + persistencia) mockeando la capa de
persistencia -- no hay Postgres real disponible en este entorno para probar
claim_next_pending/complete/fail contra las tablas reales de ai_integration."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

import gym_engine.orchestration.nodes as nodes_module
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
from gym_engine.worker.poller import _initial_state


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
    def __init__(self) -> None:
        self.completed: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []

    def complete(self, conn: object, claimed: ClaimedGeneration, **kwargs: object) -> None:
        self.completed.append({"claimed": claimed, **kwargs})

    def fail(self, conn: object, claimed: ClaimedGeneration, **kwargs: object) -> bool:
        self.failed.append({"claimed": claimed, **kwargs})
        return kwargs.get("max_attempts") == 1


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
    assert output["dias"][0]["ejercicios"][0]["ejercicio_id"] == str(ejercicio_id)


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
