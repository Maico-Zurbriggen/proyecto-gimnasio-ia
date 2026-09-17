"""Worker durable (AGENTS.md): procesa solicitudes fuera de la peticion HTTP.

Reclama filas 'PENDIENTE' (o 'PROCESANDO' con lease vencida) de
ai_integration.ai_generation_requests via repository.claim_next_pending -- FOR UPDATE SKIP
LOCKED + lease, atomico -- y corre el grafo
reducido. Si el proceso se reinicia a mitad de un intento, la lease vence sola y otro tick la
retoma; no hace falta heartbeat porque el claim ya lo resuelve (a diferencia de la version anterior
de este poller, que pollaba una tabla que no correspondia al esquema real).
"""

from __future__ import annotations

import logging
import os
import socket
import time

import psycopg
from psycopg.rows import DictRow

from gym_engine.config import Settings, get_settings
from gym_engine.llm.client import build_llm_client
from gym_engine.llm.schemas import EjercicioRef, MinimizedContext, ParametrosRutina
from gym_engine.orchestration.graph import build_graph
from gym_engine.orchestration.nodes import NodeDeps
from gym_engine.orchestration.state import GraphState
from gym_engine.persistence import repository
from gym_engine.persistence.db import get_connection
from gym_engine.persistence.models import ClaimedGeneration

logger = logging.getLogger(__name__)

_WORKER_ID = f"poller:{socket.gethostname()}:{os.getpid()}"


def _initial_state(claimed: ClaimedGeneration) -> GraphState:
    context = claimed.minimized_context
    return {
        "texto_libre": context.get("texto_libre"),
        "parametros": (
            ParametrosRutina.model_validate(context["parametros"])
            if context.get("parametros")
            else None
        ),
        "catalogo_prefiltrado": [
            EjercicioRef.model_validate(e) for e in context.get("catalogo_prefiltrado", [])
        ],
        "contexto_minimizado": MinimizedContext.model_validate(context["contexto_minimizado"]),
        "violaciones": [],
    }


def process_one(conn: psycopg.Connection[DictRow], settings: Settings) -> bool:
    claimed = repository.claim_next_pending(
        conn,
        worker_id=_WORKER_ID,
        lease_seconds=settings.generation_timeout_seconds + 30,
        contract_version=settings.contract_version,
        model_version=settings.model_version,
        configuration_version=settings.configuration_version,
    )
    if claimed is None:
        return False

    deps = NodeDeps(
        client=build_llm_client(settings), conn=conn, settings=settings, claimed=claimed
    )
    graph = build_graph(deps)
    try:
        graph.invoke(_initial_state(claimed))
        conn.commit()
    except Exception:
        conn.rollback()
        repository.fail(
            conn,
            claimed,
            attempt_state="FALLIDO",
            error_code="unexpected_error",
            max_attempts=deps.max_attempts,
        )
        conn.commit()
        logger.exception("Fallo procesando request_id=%s", claimed.request_id)
    return True


def run_forever(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    logger.info("Poller iniciado, intervalo=%ss", settings.poller_interval_seconds)
    while True:
        conn = get_connection(settings)
        try:
            processed = process_one(conn, settings)
        finally:
            conn.close()
        if not processed:
            time.sleep(settings.poller_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
