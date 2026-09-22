import uuid

from gym_engine.llm.schemas import (
    DiaRutinaCandidato,
    EjercicioRef,
    EjercicioRutinaCandidato,
    ParametrosRutina,
    RutinaEstructurada,
    SeriePrescriptaCandidata,
)
from gym_engine.orchestration import nodes


def _parametros(**overrides: object) -> ParametrosRutina:
    base = {"objetivo": "hipertrofia", "frecuencia_semanal": 3, "duracion_minutos": 60}
    base.update(overrides)
    return ParametrosRutina.model_validate(base)


def _ejercicio_ref(ejercicio_id: uuid.UUID) -> EjercicioRef:
    return EjercicioRef(id=ejercicio_id, nombre="Sentadilla", patron_movimiento="squat")


def test_validar_parametros_sin_parametros() -> None:
    result = nodes.validar_parametros({})
    assert result["violaciones"]


def test_validar_parametros_objetivo_vacio() -> None:
    parametros = ParametrosRutina.model_construct(
        objetivo="", frecuencia_semanal=3, duracion_minutos=45, restricciones=[], confianza=1.0
    )
    result = nodes.validar_parametros({"parametros": parametros})
    assert "objetivo vacio" in result["violaciones"]


def test_validar_parametros_validos() -> None:
    result = nodes.validar_parametros({"parametros": _parametros()})
    assert result["violaciones"] == []


def test_prefiltrar_catalogo_vacio() -> None:
    result = nodes.prefiltrar_catalogo({"catalogo_prefiltrado": []})
    assert result["violaciones"]


def test_prefiltrar_catalogo_con_ejercicios() -> None:
    catalogo = [EjercicioRef(id=uuid.uuid4(), nombre="Sentadilla", patron_movimiento="squat")]
    assert nodes.prefiltrar_catalogo({"catalogo_prefiltrado": catalogo}) == {}


def test_armar_contexto_ausente() -> None:
    result = nodes.armar_contexto({})
    assert result["violaciones"]


def _estructura_con(
    ejercicio_id: uuid.UUID, rep_min: int = 8, rep_max: int = 12
) -> RutinaEstructurada:
    return RutinaEstructurada(
        dias=[
            DiaRutinaCandidato(
                orden=1,
                nombre="Dia 1",
                ejercicios=[
                    EjercicioRutinaCandidato(
                        ejercicio_id=ejercicio_id,
                        orden=1,
                        series=[
                            SeriePrescriptaCandidata(
                                orden=1,
                                repeticiones_min=rep_min,
                                repeticiones_max=rep_max,
                                descanso_segundos=60,
                            )
                        ],
                    )
                ],
            )
        ]
    )


def test_validar_estructura_ejercicio_fuera_de_catalogo() -> None:
    catalogo_id = uuid.uuid4()
    ajeno_id = uuid.uuid4()
    state = {
        "catalogo_prefiltrado": [_ejercicio_ref(catalogo_id)],
        "estructura_candidata": _estructura_con(ajeno_id),
    }
    result = nodes.validar_estructura(state)
    assert any("no pertenece al catalogo" in v for v in result["violaciones"])


def test_validar_estructura_repeticiones_invertidas() -> None:
    catalogo_id = uuid.uuid4()
    state = {
        "catalogo_prefiltrado": [_ejercicio_ref(catalogo_id)],
        "estructura_candidata": _estructura_con(catalogo_id, rep_min=12, rep_max=8),
    }
    result = nodes.validar_estructura(state)
    assert any("repeticiones_min mayor que repeticiones_max" in v for v in result["violaciones"])


def test_validar_estructura_valida() -> None:
    catalogo_id = uuid.uuid4()
    state = {
        "catalogo_prefiltrado": [_ejercicio_ref(catalogo_id)],
        "estructura_candidata": _estructura_con(catalogo_id),
    }
    result = nodes.validar_estructura(state)
    assert result["violaciones"] == []
