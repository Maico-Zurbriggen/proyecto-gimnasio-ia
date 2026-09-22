"""Consumer de Vercel Queues (AGENTS.md): procesa solicitudes fuera de la peticion HTTP.

En Vercel corre en push mode: pyproject.toml declara este modulo bajo
[[tool.vercel.subscribers]], Vercel genera la funcion privada air-gapped y la invoca por
cada mensaje publicado en ROUTINE_GENERATIONS_TOPIC (ver routes.py). Fuera de Vercel
(local, self-hosted) worker/poller.py corre el mismo handler en poll mode.

El mensaje solo trae el request_id -- Postgres sigue siendo la unica fuente de verdad de
intentos/lease: claim_by_id reclama la fila puntual (mismo FOR UPDATE SKIP LOCKED + lease
que el poller anterior) y corre el grafo reducido. El reintento unico
(generation_max_retries) lo decide repository.fail() como siempre; si la solicitud no se
agoto, este consumer levanta RetryAfter para que Vercel redeliver el mismo mensaje y se
reclame de nuevo -- no se publica un mensaje nuevo ni se apoya en el redelivery ciego de
Vercel para decidir cuantos intentos hacer.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid

from vercel.queue import Message, RetryAfter, Topic, subscribe

from gym_engine.config import CONTRACT_VERSION, get_settings
from gym_engine.llm.client import build_llm_client
from gym_engine.llm.schemas import EjercicioRef, MinimizedContext, ParametrosRutina
from gym_engine.orchestration.graph import build_graph
from gym_engine.orchestration.nodes import NodeDeps
from gym_engine.orchestration.state import GraphState
from gym_engine.persistence import repository
from gym_engine.persistence.db import get_connection
from gym_engine.persistence.models import ClaimedGeneration

logger = logging.getLogger(__name__)

_WORKER_ID = f"queue-consumer:{socket.gethostname()}:{os.getpid()}"

ROUTINE_GENERATIONS_TOPIC = Topic[dict[str, str]]("routine-generations")


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


@subscribe(
    topic=ROUTINE_GENERATIONS_TOPIC,
    consumer_group="routine-generation-worker",
    max_concurrency=1,
)
def handle_routine_generation(message: Message[dict[str, str]]) -> None:
    settings = get_settings()
    request_id = uuid.UUID(message.payload["request_id"])
    conn = get_connection(settings)
    try:
        claimed = repository.claim_by_id(
            conn,
            request_id,
            worker_id=_WORKER_ID,
            lease_seconds=settings.generation_timeout_seconds + 30,
            contract_version=CONTRACT_VERSION,
            model_version=settings.model_version,
            configuration_version=settings.configuration_version,
        )
        if claimed is None:
            conn.commit()
            return

        deps = NodeDeps(
            client=build_llm_client(settings), conn=conn, settings=settings, claimed=claimed
        )
        graph = build_graph(deps)
        try:
            result = graph.invoke(_initial_state(claimed))
            conn.commit()
        except Exception:
            conn.rollback()
            exhausted = repository.fail(
                conn,
                claimed,
                attempt_state="FALLIDO",
                error_code="unexpected_error",
                max_attempts=deps.max_attempts,
            )
            conn.commit()
            logger.exception("Fallo procesando request_id=%s", claimed.request_id)
            if not exhausted:
                raise RetryAfter(0) from None
            return

        if result.get("ruta") == "FALLIDA" and not result.get("agotado", False):
            raise RetryAfter(0)
    finally:
        conn.close()
