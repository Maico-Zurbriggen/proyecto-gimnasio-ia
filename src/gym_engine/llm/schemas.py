from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EjercicioRef(BaseModel):
    """Subconjunto ya prefiltrado por el backend (RN-44a-d, RN-45).

    El servicio de IA no consulta el catálogo.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID
    nombre: str
    patron_movimiento: str


class MinimizedContext(BaseModel):
    """Equivalente a minimized_context (D2/D4 §11): sin nombre, correo, teléfono ni documento."""

    model_config = ConfigDict(extra="forbid")

    nivel_experiencia: str
    dias_semanales_disponibles: int = Field(ge=1, le=7)
    objetivos_activos: list[str] = Field(default_factory=list)
    condiciones: list[str] = Field(default_factory=list)


class ParametrosRutina(BaseModel):
    """Salida estructurada de interpretar_solicitud (RF-053)."""

    model_config = ConfigDict(extra="forbid")

    objetivo: Literal[
        "fuerza",
        "hipertrofia",
        "resistencia_muscular",
        "acondicionamiento_general",
    ]
    frecuencia_semanal: int = Field(ge=1, le=7)
    duracion_minutos: int = Field(gt=0)
    restricciones: list[str] = Field(default_factory=list)
    confianza: float = Field(ge=0.0, le=1.0, default=1.0)


class SeriePrescriptaCandidata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orden: int = Field(gt=0)
    repeticiones_min: int = Field(ge=1, le=100)
    repeticiones_max: int = Field(ge=1, le=100)
    # Kilogramos. Cero representa ejercicios sin carga externa; PostgreSQL
    # conserva este campo como decimal no nulo en la prescripcion final.
    carga_sugerida: float = Field(default=0, ge=0, le=1000)
    descanso_segundos: int = Field(ge=0, le=600)
    es_calentamiento: bool = False


class EjercicioRutinaCandidato(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ejercicio_id: UUID
    orden: int = Field(gt=0)
    nota: str | None = None
    series: list[SeriePrescriptaCandidata] = Field(min_length=1)


class DiaRutinaCandidato(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orden: int = Field(gt=0)
    nombre: str
    ejercicios: list[EjercicioRutinaCandidato] = Field(min_length=1)


class RutinaEstructurada(BaseModel):
    """Salida estructurada de generar_rutina (RF-054)."""

    model_config = ConfigDict(extra="forbid")

    dias: list[DiaRutinaCandidato] = Field(min_length=1, max_length=7)
