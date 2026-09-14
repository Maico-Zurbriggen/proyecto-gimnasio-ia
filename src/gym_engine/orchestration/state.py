"""Subconjunto del estado de maquina-estados-langgraph.pdf §3 para esta primera corrida.

Sin HITL, sin checkpointer: el grafo corre de punta a punta dentro de un tick del poller.
catalogo_prefiltrado y contexto_minimizado NO se recalculan acá — llegan ya armados por el
backend en la solicitud (este servicio no tiene acceso a tablas de dominio, AGENTS.md).
"""

from typing import Literal, TypedDict
from uuid import UUID

from gym_engine.llm.schemas import (
    EjercicioRef,
    MinimizedContext,
    ParametrosRutina,
    RutinaEstructurada,
)


class GraphState(TypedDict, total=False):
    request_id: UUID
    texto_libre: str | None
    parametros: ParametrosRutina | None
    catalogo_prefiltrado: list[EjercicioRef]
    contexto_minimizado: MinimizedContext
    estructura_candidata: RutinaEstructurada | None
    violaciones: list[str]
    intentos_generacion: int
    ruta: Literal["GENERATIVA", "FALLIDA"]
    version_modelo: str
    version_prompt: str
