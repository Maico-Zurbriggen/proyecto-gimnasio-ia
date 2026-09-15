from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GenerationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID


class PrescribedSetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int = Field(ge=1)
    min_repetitions: int = Field(ge=1)
    max_repetitions: int = Field(ge=1)
    suggested_load: float | None = Field(default=None, ge=0)
    rest_seconds: int = Field(ge=0)
    warmup: bool = False


class RoutineExerciseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int = Field(ge=1)
    exercise_id: UUID
    note: str | None = Field(default=None, max_length=500)
    sets: list[PrescribedSetOutput] = Field(min_length=1)


class RoutineDayOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    dominant_pattern: Literal[
        "EMPUJE_HORIZONTAL",
        "EMPUJE_VERTICAL",
        "TRACCION_HORIZONTAL",
        "TRACCION_VERTICAL",
        "DOMINANTE_RODILLA",
        "DOMINANTE_CADERA",
        "CORE",
        "AISLAMIENTO_SUPERIOR",
        "AISLAMIENTO_INFERIOR",
    ]
    exercises: list[RoutineExerciseOutput] = Field(min_length=1)


class RoutineGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    routine_type: Literal[
        "FUERZA",
        "HIPERTROFIA",
        "RESISTENCIA_MUSCULAR",
        "ACONDICIONAMIENTO_GENERAL",
    ]
    target_weekly_frequency: int = Field(ge=1, le=7)
    days: list[RoutineDayOutput] = Field(min_length=1, max_length=7)
    uncovered_patterns: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)


class DispatchResponse(BaseModel):
    request_id: UUID
    status: Literal["queued"] = "queued"
    message_id: str | None = None
