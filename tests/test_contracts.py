from uuid import uuid4

import pytest
from pydantic import ValidationError

from gym_engine.contracts import RoutineGenerationOutput


def valid_output() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "routine_type": "FUERZA",
        "target_weekly_frequency": 1,
        "days": [
            {
                "position": 1,
                "name": "Día 1",
                "dominant_pattern": "EMPUJE_HORIZONTAL",
                "exercises": [
                    {
                        "position": 1,
                        "exercise_id": str(uuid4()),
                        "sets": [
                            {
                                "position": 1,
                                "min_repetitions": 5,
                                "max_repetitions": 8,
                                "suggested_load": None,
                                "rest_seconds": 120,
                                "warmup": False,
                            }
                        ],
                    }
                ],
            }
        ],
        "uncovered_patterns": [],
        "explanation": "Prioriza fuerza con el catálogo permitido.",
    }


def test_accepts_versioned_structured_output() -> None:
    parsed = RoutineGenerationOutput.model_validate(valid_output())

    assert parsed.schema_version == "1.0"
    assert parsed.days[0].exercises[0].sets[0].rest_seconds == 120


def test_rejects_unknown_fields() -> None:
    output = valid_output()
    output["student_name"] = "dato no permitido"

    with pytest.raises(ValidationError):
        RoutineGenerationOutput.model_validate(output)
