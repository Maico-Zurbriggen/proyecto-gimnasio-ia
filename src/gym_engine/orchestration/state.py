"""Subconjunto del estado de maquina-estados-langgraph.pdf §3 para esta primera corrida.

Sin HITL, sin checkpointer: el grafo corre de punta a punta dentro de un claim (una fila de
ai_generation_attempts). catalogo_prefiltrado y contexto_minimizado NO se recalculan acá --
llegan ya armados por el backend en la solicitud (este servicio no tiene acceso a tablas de
dominio, AGENTS.md). Sin loop de reintento interno: reintentar es responsabilidad del consumer
de Vercel Queues (queue_consumer.py), que levanta RetryAfter para que el mismo mensaje se
redeliver y reclame la solicitud de nuevo si repository.fail() no la agotó (ver campo `agotado`).
"""

from typing import Literal, TypedDict

from gym_engine.llm.schemas import (
    EjercicioRef,
    MinimizedContext,
    ParametrosRutina,
    RutinaEstructurada,
)
from gym_engine.persistence.models import AttemptState


class GraphState(TypedDict, total=False):
    texto_libre: str | None
    parametros: ParametrosRutina | None
    catalogo_prefiltrado: list[EjercicioRef]
    contexto_minimizado: MinimizedContext
    estructura_candidata: RutinaEstructurada | None
    violaciones: list[str]
    attempt_state: AttemptState
    ruta: Literal["GENERATIVA", "FALLIDA"]
    agotado: bool
