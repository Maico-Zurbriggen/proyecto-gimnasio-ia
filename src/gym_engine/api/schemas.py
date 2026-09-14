from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from gym_engine.llm.schemas import EjercicioRef, MinimizedContext, ParametrosRutina


class RoutineGenerationCreate(BaseModel):
    """Contrato del backend -> servicio IA. catalogo_prefiltrado y contexto_minimizado ya
    vienen armados por el backend (RN-44a-d, RN-45, minimizacion de PII): este servicio no
    tiene acceso a las tablas de dominio para calcularlos."""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str = Field(min_length=1)
    gym_id: UUID
    student_id: UUID
    requested_by_user_id: UUID
    texto_libre: str | None = None
    parametros: ParametrosRutina | None = None
    catalogo_prefiltrado: list[EjercicioRef] = Field(min_length=1)
    contexto_minimizado: MinimizedContext


class RoutineGenerationAccepted(BaseModel):
    request_id: UUID
    status: str


class RoutineGenerationStatus(BaseModel):
    request_id: UUID
    status: str
    estructura_candidata: dict[str, Any] | None = None
    violaciones: list[str] | None = None
    error: str | None = None
