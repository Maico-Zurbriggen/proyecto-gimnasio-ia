"""Tipos que reflejan el esquema real de ai_integration en Neon (verificado contra la BD:
schemas ai_integration.ai_generation_requests/_attempts/_results/_result_validations).

No hay ORM acá a propósito: este repo no migra ese esquema (backend es dueño), así que
mapear con SQLAlchemy es una capa que se puede desincronizar en silencio. Todo el acceso
es SQL explícito en repository.py contra las columnas reales.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

# ai_integration."AiGenerationRequestState"
RequestState = Literal["PENDIENTE", "PROCESANDO", "COMPLETADA", "NO_DISPONIBLE", "CANCELADA"]

# ai_integration."AiGenerationAttemptState"
AttemptState = Literal[
    "PENDIENTE", "PROCESANDO", "COMPLETADO", "FALLIDO", "AGOTADO_POR_TIEMPO", "SALIDA_INVALIDA"
]


@dataclass(frozen=True)
class ClaimedGeneration:
    request_id: uuid.UUID
    attempt_id: uuid.UUID
    attempt_number: int
    minimized_context: dict[str, Any]
    preferences: dict[str, Any]
    retention_until: datetime


@dataclass(frozen=True)
class RequestStatus:
    request_id: uuid.UUID
    state: RequestState
    structured_output: dict[str, Any] | None
    error_code: str | None
