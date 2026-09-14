"""Worker durable (AGENTS.md): procesa solicitudes fuera de la peticion HTTP.

Sin cola dedicada en esta corrida: reclama filas 'pending' de ai_generation_requests
(claim atomico en repository.claim_next_pending) y corre el grafo reducido. Si el proceso
se reinicia, las filas 'pending' siguen ahi y se retoman; una fila que quedo 'processing'
por un crash a mitad de camino NO se reintenta automaticamente en esta version (limitacion
conocida, documentada en la spec, pendiente de lease/heartbeat en una proxima corrida).
"""

from __future__ import annotations

import logging
import time

from sqlalchemy.orm import Session, sessionmaker

from gym_engine.config import Settings, get_settings
from gym_engine.llm.client import OllamaClient
from gym_engine.llm.schemas import EjercicioRef, MinimizedContext, ParametrosRutina
from gym_engine.orchestration.graph import build_graph
from gym_engine.orchestration.nodes import NodeDeps
from gym_engine.orchestration.state import GraphState
from gym_engine.persistence import repository
from gym_engine.persistence.db import get_engine, get_session_factory
from gym_engine.persistence.models import AiGenerationRequest

logger = logging.getLogger(__name__)


def _initial_state(request: AiGenerationRequest) -> GraphState:
    return {
        "request_id": request.id,
        "texto_libre": request.texto_libre,
        "parametros": (
            ParametrosRutina.model_validate(request.parametros) if request.parametros else None
        ),
        "catalogo_prefiltrado": [
            EjercicioRef.model_validate(e) for e in request.catalogo_prefiltrado
        ],
        "contexto_minimizado": MinimizedContext.model_validate(request.contexto_minimizado),
        "violaciones": [],
        "intentos_generacion": 0,
    }


def process_one(session: Session, settings: Settings) -> bool:
    request = repository.claim_next_pending(session)
    if request is None:
        return False

    deps = NodeDeps(client=OllamaClient(settings), session=session, settings=settings)
    graph = build_graph(deps)
    try:
        graph.invoke(_initial_state(request))
        session.commit()
    except Exception:
        session.rollback()
        repository.mark_failed(
            session, request_id=request.id, error="fallo no controlado en el grafo"
        )
        session.commit()
        logger.exception("Fallo procesando request_id=%s", request.id)
    return True


def run_forever(
    session_factory: sessionmaker[Session] | None = None, settings: Settings | None = None
) -> None:
    settings = settings or get_settings()
    factory = session_factory or get_session_factory(get_engine(settings))
    logger.info("Poller iniciado, intervalo=%ss", settings.poller_interval_seconds)
    while True:
        session = factory()
        try:
            processed = process_one(session, settings)
        finally:
            session.close()
        if not processed:
            time.sleep(settings.poller_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
