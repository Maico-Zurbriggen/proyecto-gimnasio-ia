"""Nodos del grafo reducido (sin HITL, sin tool-calling, sin checkpointer en esta corrida).

prefiltrar_catalogo y armar_contexto ya no consultan nada: sólo validan lo que llegó en la
solicitud, porque el servicio de IA no tiene acceso a tablas de dominio (AGENTS.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from gym_engine.config import Settings
from gym_engine.llm.client import LlmInvalidOutput, LlmUnavailable, OllamaClient
from gym_engine.llm.schemas import ParametrosRutina, RutinaEstructurada
from gym_engine.orchestration.state import GraphState
from gym_engine.persistence import repository

VERSION_PROMPT = "generative/generar-rutina@1"

NodeFn = Callable[[GraphState], dict[str, Any]]
RouterFn = Callable[[GraphState], str]


@dataclass
class NodeDeps:
    client: OllamaClient
    session: Session
    settings: Settings

    @property
    def max_intentos(self) -> int:
        return self.settings.generation_max_retries + 1


def entry_router(state: GraphState) -> str:
    if state.get("texto_libre") and not state.get("parametros"):
        return "interpretar_solicitud"
    return "validar_parametros"


def build_interpretar_solicitud(deps: NodeDeps) -> NodeFn:
    def interpretar_solicitud(state: GraphState) -> dict[str, Any]:
        attempt = repository.record_attempt(
            deps.session,
            request_id=state["request_id"],
            node="interpretar_solicitud",
            attempt_number=1,
        )
        prompt = (
            "Interpretá el siguiente pedido de rutina en lenguaje natural y devolvé los "
            f"parámetros estructurados. Pedido: {state['texto_libre']}"
        )
        try:
            parametros = deps.client.generate_structured(prompt, ParametrosRutina)
        except (LlmUnavailable, LlmInvalidOutput) as exc:
            repository.finish_attempt(deps.session, attempt, error=str(exc))
            return {"ruta": "FALLIDA", "violaciones": [f"interpretar_solicitud: {exc}"]}
        repository.finish_attempt(deps.session, attempt)
        return {"parametros": parametros}

    return interpretar_solicitud


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


def build_generar_rutina(deps: NodeDeps) -> NodeFn:
    def generar_rutina(state: GraphState) -> dict[str, Any]:
        intentos = state.get("intentos_generacion", 0) + 1
        attempt = repository.record_attempt(
            deps.session,
            request_id=state["request_id"],
            node="generar_rutina",
            attempt_number=intentos,
        )
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
        except (LlmUnavailable, LlmInvalidOutput) as exc:
            repository.finish_attempt(deps.session, attempt, error=str(exc))
            return {"intentos_generacion": intentos, "violaciones": [f"generar_rutina: {exc}"]}
        repository.finish_attempt(deps.session, attempt)
        return {
            "intentos_generacion": intentos,
            "estructura_candidata": estructura,
            "version_modelo": deps.settings.ollama_model,
            "version_prompt": VERSION_PROMPT,
            "violaciones": [],
        }

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


def build_route_after_validation(deps: NodeDeps) -> RouterFn:
    def route_after_validation(state: GraphState) -> str:
        if not state.get("violaciones"):
            return "persistir_resultado"
        if state.get("intentos_generacion", 0) < deps.max_intentos:
            return "generar_rutina"
        return "via_fallida"

    return route_after_validation


def build_persistir_resultado(deps: NodeDeps) -> NodeFn:
    def persistir_resultado(state: GraphState) -> dict[str, Any]:
        estructura = state["estructura_candidata"]
        assert estructura is not None
        repository.save_validation(
            deps.session, request_id=state["request_id"], valid=True, violations=[]
        )
        repository.save_result(
            deps.session,
            request_id=state["request_id"],
            estructura_candidata=estructura.model_dump(mode="json"),
            version_modelo=state["version_modelo"],
            version_prompt=state["version_prompt"],
        )
        return {}

    return persistir_resultado


def build_via_fallida(deps: NodeDeps) -> NodeFn:
    def via_fallida(state: GraphState) -> dict[str, Any]:
        violaciones = state.get("violaciones") or ["fallo no especificado"]
        repository.save_validation(
            deps.session,
            request_id=state["request_id"],
            valid=False,
            violations=violaciones,
        )
        repository.mark_failed(
            deps.session, request_id=state["request_id"], error="; ".join(violaciones)
        )
        return {"ruta": "FALLIDA"}

    return via_fallida


def route_after_interpretar(state: GraphState) -> str:
    if state.get("ruta") == "FALLIDA":
        return "via_fallida"
    return "validar_parametros"


def early_exit_router(state: GraphState) -> str:
    """Usado tras validar_parametros / prefiltrar_catalogo / armar_contexto."""
    if state.get("violaciones"):
        return "via_fallida"
    return "continue"


__all__ = [
    "NodeDeps",
    "entry_router",
    "route_after_interpretar",
    "build_interpretar_solicitud",
    "validar_parametros",
    "prefiltrar_catalogo",
    "armar_contexto",
    "build_generar_rutina",
    "validar_estructura",
    "build_route_after_validation",
    "build_persistir_resultado",
    "build_via_fallida",
    "early_exit_router",
]
