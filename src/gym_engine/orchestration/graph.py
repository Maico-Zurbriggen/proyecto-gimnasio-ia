from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from gym_engine.orchestration import nodes
from gym_engine.orchestration.nodes import NodeDeps
from gym_engine.orchestration.state import GraphState


def build_graph(deps: NodeDeps) -> CompiledStateGraph:
    """Subconjunto de maquina-estados-langgraph.pdf: sin HITL, sin checkpointer, sin tools.

    Corre de punta a punta dentro de un tick del poller (worker/poller.py).
    """
    graph = StateGraph(GraphState)

    graph.add_node("interpretar_solicitud", nodes.build_interpretar_solicitud(deps))
    graph.add_node("validar_parametros", nodes.validar_parametros)
    graph.add_node("prefiltrar_catalogo", nodes.prefiltrar_catalogo)
    graph.add_node("armar_contexto", nodes.armar_contexto)
    graph.add_node("generar_rutina", nodes.build_generar_rutina(deps))
    graph.add_node("validar_estructura", nodes.validar_estructura)
    graph.add_node("persistir_resultado", nodes.build_persistir_resultado(deps))
    graph.add_node("via_fallida", nodes.build_via_fallida(deps))

    graph.set_conditional_entry_point(
        nodes.entry_router,
        {
            "interpretar_solicitud": "interpretar_solicitud",
            "validar_parametros": "validar_parametros",
        },
    )

    graph.add_conditional_edges(
        "interpretar_solicitud",
        nodes.route_after_interpretar,
        {"via_fallida": "via_fallida", "validar_parametros": "validar_parametros"},
    )
    graph.add_conditional_edges(
        "validar_parametros",
        nodes.early_exit_router,
        {"via_fallida": "via_fallida", "continue": "prefiltrar_catalogo"},
    )
    graph.add_conditional_edges(
        "prefiltrar_catalogo",
        nodes.early_exit_router,
        {"via_fallida": "via_fallida", "continue": "armar_contexto"},
    )
    graph.add_conditional_edges(
        "armar_contexto",
        nodes.early_exit_router,
        {"via_fallida": "via_fallida", "continue": "generar_rutina"},
    )
    graph.add_edge("generar_rutina", "validar_estructura")
    graph.add_conditional_edges(
        "validar_estructura",
        nodes.route_after_validation,
        {"persistir_resultado": "persistir_resultado", "via_fallida": "via_fallida"},
    )
    graph.add_edge("persistir_resultado", END)
    graph.add_edge("via_fallida", END)

    return graph.compile()
