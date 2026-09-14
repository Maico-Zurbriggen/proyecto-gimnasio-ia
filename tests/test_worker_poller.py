import uuid

from sqlalchemy.orm import Session

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
from gym_engine.persistence import repository
from gym_engine.persistence.models import AiGenerationResult
from gym_engine.worker.poller import _initial_state


class FakeOllamaClient:
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


def _settings() -> Settings:
    return Settings(database_url="sqlite://", ollama_model="test-model")


def test_process_one_generates_and_persists_result(session: Session) -> None:
    ejercicio_id = uuid.uuid4()
    request = repository.create_request(
        session,
        idempotency_key="worker-key",
        gym_id=uuid.uuid4(),
        student_id=uuid.uuid4(),
        requested_by_user_id=uuid.uuid4(),
        catalogo_prefiltrado=[
            {"id": str(ejercicio_id), "nombre": "Sentadilla", "patron_movimiento": "squat"}
        ],
        contexto_minimizado={
            "nivel_experiencia": "intermedio",
            "dias_semanales_disponibles": 3,
        },
        parametros=ParametrosRutina(
            objetivo="fuerza", frecuencia_semanal=3, duracion_minutos=50
        ).model_dump(mode="json"),
    )
    session.commit()

    claimed = repository.claim_next_pending(session)
    assert claimed is not None

    deps = NodeDeps(
        client=FakeOllamaClient(ejercicio_id),  # type: ignore[arg-type]
        session=session,
        settings=_settings(),
    )
    graph = build_graph(deps)

    graph.invoke(_initial_state(claimed))
    session.commit()

    refreshed = repository.get_request(session, request.id)
    assert refreshed is not None
    assert refreshed.status == "completed"

    result = session.query(AiGenerationResult).filter_by(request_id=request.id).one()
    dia = result.estructura_candidata["dias"][0]
    assert dia["ejercicios"][0]["ejercicio_id"] == str(ejercicio_id)
