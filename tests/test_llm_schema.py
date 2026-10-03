from copy import deepcopy

from pytest import raises

from gym_engine.llm import CompactRoutinePlan
from gym_engine.llm_schema import build_native_generation_schema


def preferences() -> dict:
    return {
        "muscle_counts_per_day": [{"muscle": "GLUTEO", "count": 3}],
        "allowed_catalog": [{"id": str(index), "primary_muscles": ["GLUTEO"]} for index in range(9)]
        + [
            {"id": str(index + 9), "primary_muscles": [muscle]}
            for index, muscle in enumerate(
                ["DORSAL", "PECTORAL", "ABDOMINALES", "CUADRICEPS", "BICEPS"]
            )
        ],
        "prescription_constraints": {
            "purposes": {
                "HIPERTROFIA": {
                    "weekly_frequency": [3, 6],
                    "exercises_per_day": [5, 8],
                    "work_sets_per_exercise": [3, 4],
                    "repetitions": [6, 12],
                    "rest_seconds": [60, 120],
                }
            }
        },
    }


def compile_schema(data: dict | None = None, available_days: int = 3) -> dict:
    return build_native_generation_schema(
        CompactRoutinePlan.model_json_schema(),
        {"available_days_per_week": available_days},
        data or preferences(),
    )


def test_native_schema_enforces_distinct_primary_counts_and_excludes_them_from_extras() -> None:
    variant = compile_schema()["anyOf"][0]
    day = variant["properties"]["days"]["items"]
    options = day["properties"]["required_exercises"]["properties"]["GLUTEO"]["enum"]
    assert len(options) == 56
    for option in options:
        indices = [exercise["exercise"] for exercise in option]
        assert len(indices) == len(set(indices)) == 3
        assert all(index < 8 for index in indices)
    assert set(day["required"]) >= {"exercises", "required_exercises"}
    assert day["properties"]["exercises"]["items"]["properties"]["exercise"]["enum"] == list(
        range(9, 14)
    )
    assert day["properties"]["exercises"]["minItems"] == 2
    assert day["properties"]["exercises"]["maxItems"] == 5


def test_native_schema_keeps_prescriptions_in_backend_ranges_and_available_days() -> None:
    properties = compile_schema()["anyOf"][0]["properties"]
    assert properties["routine_type"] == {"const": "HIPERTROFIA"}
    assert properties["days"]["minItems"] == properties["days"]["maxItems"] == 3
    assert properties["prescription"]["minItems"] == properties["prescription"]["maxItems"] == 1
    fields = properties["prescription"]["items"]["properties"]
    assert (fields["n"]["minimum"], fields["n"]["maximum"]) == (3, 4)
    assert (fields["rest"]["minimum"], fields["rest"]["maximum"]) == (60, 120)
    assert fields["warmup"] == {"const": False}
    assert all(6 <= lower <= upper <= 12 for lower, upper in fields["reps"]["enum"])
    assert [6, 12] in fields["reps"]["enum"]


def test_native_schema_handles_zero_and_multiple_muscle_requirements() -> None:
    data = preferences()
    data["muscle_counts_per_day"].append({"muscle": "BICEPS", "count": 0})
    schema = compile_schema(data)["anyOf"][0]
    day = schema["properties"]["days"]["items"]
    focus = day["properties"]["required_exercises"]
    assert focus["required"] == ["GLUTEO", "BICEPS"]
    assert focus["properties"]["BICEPS"]["enum"] == [[]]
    assert 13 not in day["properties"]["exercises"]["items"]["properties"]["exercise"]["enum"]


def test_native_schema_adapts_grammar_without_mutating_source_contract_or_preferences() -> None:
    source = CompactRoutinePlan.model_json_schema()
    source_before = deepcopy(source)
    data = preferences()
    data_before = deepcopy(data)
    schema = build_native_generation_schema(source, {"available_days_per_week": 3}, data)
    assert source == source_before
    assert data == data_before

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            assert "$ref" not in value and "$defs" not in value and "prefixItems" not in value
            if "maxLength" in value:
                assert value["maxLength"] <= 120
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(schema)


def test_native_schema_rejects_impossible_day_availability_before_inference() -> None:
    with raises(ValueError, match="No prescription fits"):
        compile_schema(available_days=2)


def test_native_schema_rejects_insufficient_primary_choices_before_inference() -> None:
    data = preferences()
    data["muscle_counts_per_day"][0]["count"] = 8
    data["allowed_catalog"] = data["allowed_catalog"][:2]
    with raises(ValueError, match="insufficient"):
        compile_schema(data)
