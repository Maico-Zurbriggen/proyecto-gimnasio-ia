from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MovementPattern = Literal[
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
TrainingPurpose = Literal[
    "FUERZA", "HIPERTROFIA", "RESISTENCIA_MUSCULAR", "ACONDICIONAMIENTO_GENERAL"
]
DiagnosticText = Annotated[str, Field(min_length=1, max_length=1000)]


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
    dominant_pattern: MovementPattern
    exercises: list[RoutineExerciseOutput] = Field(min_length=1)


class RoutineGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    routine_type: TrainingPurpose
    target_weekly_frequency: int = Field(ge=1, le=7)
    days: list[RoutineDayOutput] = Field(min_length=1, max_length=7)
    uncovered_patterns: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)


class DispatchResponse(BaseModel):
    request_id: UUID
    status: Literal["queued"] = "queued"
    message_id: str | None = None


class PrescribedSetV2(PrescribedSetOutput):
    position: int = Field(ge=1, le=2147483647)
    min_repetitions: int = Field(ge=1, le=2147483647)
    max_repetitions: int = Field(ge=1, le=2147483647)
    rest_seconds: int = Field(ge=0, le=2147483647)
    suggested_load: float | None = Field(default=None, ge=0, le=99999999.99)


class RoutineExerciseV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: int = Field(ge=1, le=2147483647)
    exercise_id: UUID
    note: str | None = Field(default=None, max_length=500)
    sets: list[PrescribedSetV2] = Field(min_length=1, max_length=500)


class RoutineDayV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: int = Field(ge=1, le=2147483647)
    name: str = Field(min_length=1, max_length=120)
    dominant_pattern: MovementPattern
    exercises: list[RoutineExerciseV2] = Field(min_length=1, max_length=500)


class RoutineProposalV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    outcome: Literal["PROPOSED"] = "PROPOSED"
    routine_type: TrainingPurpose
    target_weekly_frequency: int = Field(ge=1, le=7)
    days: list[RoutineDayV2] = Field(min_length=1, max_length=7)
    explanation: str = Field(min_length=1, max_length=4000)
    warnings: list[DiagnosticText] = Field(max_length=50)


class RoutineUnableV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    outcome: Literal["UNABLE"] = "UNABLE"
    reason: str = Field(min_length=1, max_length=4000)
    missing_information: list[DiagnosticText] = Field(max_length=50)
