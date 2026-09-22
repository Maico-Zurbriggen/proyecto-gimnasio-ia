"""Nodos del grafo reducido (sin HITL, sin tool-calling, sin checkpointer en esta corrida).

prefiltrar_catalogo y armar_contexto no consultan nada: solo validan lo que llego en la
solicitud, porque el servicio de IA no tiene acceso a tablas de dominio (AGENTS.md).

Sin retry interno: cada invocacion del grafo corresponde a UN claim = UNA fila de
ai_generation_attempts (esquema real, ver persistence/repository.py). Si algo falla,
persistir_fallo decide si la solicitud vuelve a PENDIENTE (el consumer de Vercel Queues en
worker/queue_consumer.py levanta RetryAfter para que el mismo mensaje se redeliver y la
reclame de nuevo, abriendo un intento nuevo) o se agota (NO_DISPONIBLE) -- eso reemplaza
al loop-back generar_rutina<->validar_estructura que tenia la version anterior.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import DictRow

from gym_engine.config import Settings
from gym_engine.llm.client import LlmClient, LlmInvalidOutput, LlmUnavailable
from gym_engine.llm.schemas import ParametrosRutina, RutinaEstructurada
from gym_engine.orchestration.state import GraphState
from gym_engine.persistence import repository
from gym_engine.persistence.models import ClaimedGeneration

NodeFn = Callable[[GraphState], dict[str, Any]]
RouterFn = Callable[[GraphState], str]


@dataclass
class NodeDeps:
    client: LlmClient
    conn: psycopg.Connection[DictRow]
    settings: Settings
    claimed: ClaimedGeneration

    @property
    def max_attempts(self) -> int:
        return self.settings.generation_max_retries + 1


def entry_router(state: GraphState) -> str:
    if state.get("texto_libre") and not state.get("parametros"):
        return "interpretar_solicitud"
    return "validar_parametros"


def build_interpretar_solicitud(deps: NodeDeps) -> NodeFn:
    def interpretar_solicitud(state: GraphState) -> dict[str, Any]:
        prompt = (
            "Interpretá el siguiente pedido de rutina en lenguaje natural y devolvé los "
            f"parámetros estructurados. Pedido: {state['texto_libre']}"
        )
        try:
            parametros = deps.client.generate_structured(prompt, ParametrosRutina)
        except LlmUnavailable as exc:
            return {
                "ruta": "FALLIDA",
                "violaciones": [f"interpretar_solicitud: {exc}"],
                "attempt_state": "FALLIDO",
            }
        except LlmInvalidOutput as exc:
            return {
                "ruta": "FALLIDA",
                "violaciones": [f"interpretar_solicitud: {exc}"],
                "attempt_state": "SALIDA_INVALIDA",
            }
        return {"parametros": parametros}

    return interpretar_solicitud


def route_after_interpretar(state: GraphState) -> str:
    if state.get("ruta") == "FALLIDA":
        return "via_fallida"
    return "validar_parametros"


def validar_parametros(state: GraphState) -> dict[str, Any]:
    parametros = state.get("parametros")
    if parametros is None:
        return {"violaciones": ["faltan parametros (ni texto_libre ni parametros confirmados)"]}
    violaciones: list[str] = []
    if not (1 <= parametros.frecuencia_semanal <= 7):
        violaciones.append("frecuencia_semanal fuera de RN-38 (1-7)")
    if not parametros.objetivo:
        violaciones.append("objetivo vacio")
    return {"violaciones": violaciones}


def prefiltrar_catalogo(state: GraphState) -> dict[str, Any]:
    catalogo = state.get("catalogo_prefiltrado") or []
    if not catalogo:
        return {
            "violaciones": ["catalogo_prefiltrado vacio: el backend no envio ejercicios admisibles"]
        }
    return {}


def armar_contexto(state: GraphState) -> dict[str, Any]:
    contexto = state.get("contexto_minimizado")
    if contexto is None:
        return {"violaciones": ["contexto_minimizado ausente"]}
    return {}


def early_exit_router(state: GraphState) -> str:
    """Usado tras validar_parametros / prefiltrar_catalogo / armar_contexto."""
    if state.get("violaciones"):
        return "via_fallida"
    return "continue"


def build_generar_rutina(deps: NodeDeps) -> NodeFn:
    def generar_rutina(state: GraphState) -> dict[str, Any]:
        catalogo = state["catalogo_prefiltrado"]
        contexto = state["contexto_minimizado"]
        parametros = state["parametros"]
        assert parametros is not None, "generar_rutina requiere parametros ya validados"
        ids_disponibles = ", ".join(f"{e.id} ({e.nombre})" for e in catalogo)
        prompt = (
            "Generá una rutina de entrenamiento usando EXCLUSIVAMENTE estos ejercicios "
            f"(id y nombre): {ids_disponibles}. "
            f"Objetivo: {parametros.objetivo}. "
            f"Frecuencia semanal: {parametros.frecuencia_semanal}. "
            f"Duracion por sesion (min): {parametros.duracion_minutos}. "
            f"Restricciones: {parametros.restricciones}. "
            f"Contexto del alumno: nivel={contexto.nivel_experiencia}, "
            f"dias_disponibles={contexto.dias_semanales_disponibles}, "
            f"condiciones={contexto.condiciones}."
        )
        try:
            estructura = deps.client.generate_structured(prompt, RutinaEstructurada)
        except LlmUnavailable as exc:
            return {"violaciones": [f"generar_rutina: {exc}"], "attempt_state": "FALLIDO"}
        except LlmInvalidOutput as exc:
            return {
                "violaciones": [f"generar_rutina: {exc}"],
                "attempt_state": "SALIDA_INVALIDA",
            }
        return {"estructura_candidata": estructura, "violaciones": []}

    return generar_rutina


def validar_estructura(state: GraphState) -> dict[str, Any]:
    estructura = state.get("estructura_candidata")
    if estructura is None:
        return {"violaciones": state.get("violaciones") or ["sin estructura candidata"]}

    ids_validos = {ejercicio.id for ejercicio in state["catalogo_prefiltrado"]}
    violaciones: list[str] = []
    for dia in estructura.dias:
        for ejercicio in dia.ejercicios:
            if ejercicio.ejercicio_id not in ids_validos:
                violaciones.append(
                    f"dia {dia.orden}: ejercicio {ejercicio.ejercicio_id} "
                    "no pertenece al catalogo prefiltrado"
                )
            for serie in ejercicio.series:
                if serie.repeticiones_min > serie.repeticiones_max:
                    violaciones.append(
                        f"dia {dia.orden} ejercicio {ejercicio.ejercicio_id}: "
                        "repeticiones_min mayor que repeticiones_max"
                    )
    return {"violaciones": violaciones}


def route_after_validation(state: GraphState) -> str:
    if not state.get("violaciones"):
        return "persistir_resultado"
    return "via_fallida"


def build_persistir_resultado(deps: NodeDeps) -> NodeFn:
    def persistir_resultado(state: GraphState) -> dict[str, Any]:
        estructura = state["estructura_candidata"]
        assert estructura is not None
        output = estructura.model_dump(mode="json")
        canonical = json.dumps(
            output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        output_hash = hashlib.sha256(canonical).hexdigest()
        repository.complete(
            deps.conn,
            deps.claimed,
            output=output,
            output_hash=output_hash,
            model_version=deps.settings.model_version,
            configuration_version=deps.settings.configuration_version,
        )
        return {}

    return persistir_resultado


def build_via_fallida(deps: NodeDeps) -> NodeFn:
    def via_fallida(state: GraphState) -> dict[str, Any]:
        violaciones = state.get("violaciones") or ["fallo no especificado"]
        attempt_state = state.get("attempt_state") or "SALIDA_INVALIDA"
        error_code = "; ".join(violaciones)
        exhausted = repository.fail(
            deps.conn,
            deps.claimed,
            attempt_state=attempt_state,
            error_code=error_code,
            max_attempts=deps.max_attempts,
        )
        return {"ruta": "FALLIDA", "agotado": exhausted}

    return via_fallida


__all__ = [
    "NodeDeps",
    "entry_router",
    "route_after_interpretar",
    "build_interpretar_solicitud",
    "validar_parametros",
    "prefiltrar_catalogo",
    "armar_contexto",
    "early_exit_router",
    "build_generar_rutina",
    "validar_estructura",
    "route_after_validation",
    "build_persistir_resultado",
    "build_via_fallida",
]
