from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
from pytest import MonkeyPatch, mark, raises

import gym_engine.llm as llm_module
from gym_engine.llm import CompactRoutinePlan, InvalidPrescriptionError, OllamaClient, RoutinePlan


@mark.parametrize("retry", [False, True])
def test_ollama_client_uses_bearer_and_passes_minimized_context_and_catalog(
    monkeypatch: MonkeyPatch,
    retry: bool,
) -> None:
    exercise_id = str(uuid4())
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        response = httpx.Response(
            200,
            json={
                "model": "qwen3.5:9b",
                "done": True,
                "message": {
                    "content": json.dumps(
                        {
                            "schema_version": "1.0",
                            "routine_type": "FUERZA",
                            "prescription": [{"n": 3, "reps": [5, 8], "rest": 120}],
                            "days": [
                                {
                                    "name": "Día 1",
                                    "dominant_pattern": "DOMINANTE_RODILLA",
                                    "required_exercises": {"CUADRICEPS": [{"exercise": 0}]},
                                    "exercises": [],
                                }
                            ],
                            "uncovered_patterns": [],
                            "explanation": "Prioriza fuerza con el catálogo disponible.",
                        }
                    )
                },
            },
        )
        envelope = response.json()
        content = envelope["message"]["content"]
        split_at = len(content) // 2
        chunks = [
            {"message": {"content": content[:split_at]}, "done": False},
            {
                "model": envelope["model"],
                "message": {"content": content[split_at:]},
                "done": True,
            },
        ]
        return httpx.Response(200, content="\n".join(json.dumps(chunk) for chunk in chunks))

    original_async_client = httpx.AsyncClient

    def mock_async_client(**kwargs: object) -> httpx.AsyncClient:
        return original_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", mock_async_client)
    client = OllamaClient(
        base_url="https://polo.example.invalid/polo",
        model="qwen3.5:9b",
        api_token="opaque-token",
        timeout_seconds=30,
    )

    async def run_client() -> None:
        await client.ping()
        result = await client.generate_routine(
            {
                "experience_level": "intermedio",
                "available_days_per_week": 3,
                "active_goals": ["fuerza"],
                "conditions": [],
            },
            {
                "free_text": "Quiero ganar fuerza",
                "local_test_regeneration": {"replaces_proposed_routine_id": "internal-routine-id"},
                "parameters": None,
                "muscle_counts_per_day": [{"muscle": "CUADRICEPS", "count": 1}],
                "allowed_catalog": [
                    {
                        "id": exercise_id,
                        "name": "Sentadilla",
                        "movement_pattern": "DOMINANTE_RODILLA",
                        "primary_muscles": ["CUADRICEPS"],
                    },
                    {
                        "id": str(uuid4()),
                        "name": "Otro ejercicio",
                        "movement_pattern": "DOMINANTE_CADERA",
                        "primary_muscles": ["GLUTEO"],
                        "secondary_muscles": ["CUADRICEPS"],
                    },
                ],
            },
            retry=retry,
        )
        assert str(result.output.days[0].exercises[0].exercise_id) == exercise_id
        sets = result.output.days[0].exercises[0].sets
        assert [prescribed.position for prescribed in sets] == [1, 2, 3]
        assert all(prescribed.min_repetitions == 5 for prescribed in sets)
        assert all(prescribed.max_repetitions == 8 for prescribed in sets)
        assert all(prescribed.rest_seconds == 120 for prescribed in sets)
        assert all(prescribed.suggested_load is None for prescribed in sets)

    asyncio.run(run_client())

    assert len(seen) == 2
    assert {request.headers["authorization"] for request in seen} == {"Bearer opaque-token"}
    assert {request.headers["ngrok-skip-browser-warning"] for request in seen} == {"1"}
    assert seen[0].url.path == "/polo/api/tags"
    assert seen[1].url.path == "/polo/api/chat"
    prompt = json.loads(json.loads(seen[1].content)["messages"][1]["content"])
    assert prompt["preferences"]["allowed_catalog"][0]["index"] == 0
    assert "id" not in prompt["preferences"]["allowed_catalog"][0]
    assert exercise_id not in str(seen[1].content)
    assert "local_test_regeneration" not in prompt["preferences"]
    assert "internal-routine-id" not in str(seen[1].content)
    assert "schema_version" in prompt["output_schema"]["properties"]
    native_schema = json.loads(seen[1].content)["format"]
    assert native_schema["properties"]["days"]["items"]["properties"]["required_exercises"][
        "properties"
    ]["CUADRICEPS"]["enum"] == [[{"exercise": 0}]]
    assert prompt["output_schema"]["$defs"]["CompactExercisePlan"]["properties"]["exercise"][
        "enum"
    ] == [0, 1]
    assert json.loads(seen[1].content)["think"] is False
    assert json.loads(seen[1].content)["stream"] is True
    assert "no lo sustituyas" in json.loads(seen[1].content)["messages"][0]["content"]
    assert prompt["preferences"]["mandatory_muscle_choices"] == [
        {"muscle": "CUADRICEPS", "count": 1, "allowed_indices": [0]},
    ]
    assert (
        "El primer intento fue rechazado" in json.loads(seen[1].content)["messages"][0]["content"]
    ) == retry


@mark.parametrize("index", [1, 100])
def test_compact_plan_rejects_unknown_indices(index: int) -> None:
    compact = CompactRoutinePlan.model_validate(
        {
            "schema_version": "1.0",
            "routine_type": "FUERZA",
            "prescription": [{"n": 3, "reps": [5, 8], "rest": 120}],
            "days": [
                {
                    "name": "Día 1",
                    "dominant_pattern": "CORE",
                    "exercises": [
                        {
                            "exercise": index,
                            "groups": [{"n": 3, "reps": [5, 8], "rest": 120}],
                        }
                    ],
                }
            ],
            "explanation": "Candidata.",
        }
    )
    with raises(InvalidPrescriptionError, match="Unknown"):
        compact.to_plan([{"id": str(uuid4())}])


@mark.parametrize(
    "exercise",
    [
        {"exercise": -1, "groups": [{"n": 3, "reps": [5, 8], "rest": 120}]},
        {"exercise": 0, "groups": []},
        {"exercise": 0, "groups": [{"n": 0, "reps": [5, 8], "rest": 120}]},
        {"exercise": 0, "groups": [{"n": 3, "reps": [5, 8], "rest": -1}]},
    ],
)
def test_compact_plan_rejects_invalid_indices_and_groups(exercise: dict[str, object]) -> None:
    with raises(ValueError):
        CompactRoutinePlan.model_validate(
            {
                "schema_version": "1.0",
                "routine_type": "FUERZA",
                "prescription": [{"n": 3, "reps": [5, 8], "rest": 120}],
                "days": [{"name": "Día 1", "dominant_pattern": "CORE", "exercises": [exercise]}],
                "explanation": "Candidata.",
            }
        )


def test_compact_plan_expands_selected_groups_without_changing_prescriptions() -> None:
    identifiers = [str(uuid4()), str(uuid4())]
    compact = CompactRoutinePlan.model_validate(
        {
            "schema_version": "1.0",
            "routine_type": "FUERZA",
            "prescription": [{"n": 4, "reps": [6, 8], "rest": 180}],
            "days": [
                {
                    "name": "Día 1",
                    "dominant_pattern": "CORE",
                    "required_exercises": {
                        "GLUTEO": [
                            {
                                "exercise": 1,
                                "groups": [
                                    {"n": 1, "reps": [10, 10], "rest": 60, "warmup": True},
                                    {"n": 3, "reps": [5, 8], "rest": 120},
                                ],
                                "note": "Controlar la bajada.",
                            },
                        ]
                    },
                    "exercises": [{"exercise": 0}],
                }
            ],
            "explanation": "Candidata.",
        }
    )
    plan = compact.to_plan([{"id": identifier} for identifier in identifiers])
    output = plan.to_output()
    assert output.target_weekly_frequency == 1
    assert output.days[0].exercises[0].note == "Controlar la bajada."
    assert [str(exercise.exercise_id) for exercise in output.days[0].exercises] == identifiers[::-1]
    assert [exercise.position for exercise in output.days[0].exercises] == [1, 2]
    assert [item.warmup for item in output.days[0].exercises[0].sets] == [True, False, False, False]
    assert all(item.rest_seconds == 180 for item in output.days[0].exercises[1].sets)
    assert all(
        item.suggested_load is None
        for exercise in output.days[0].exercises
        for item in exercise.sets
    )


def test_explicit_primary_muscle_counts_are_checked_in_every_day() -> None:
    triceps_ids = [str(uuid4()) for _ in range(3)]
    press_id = str(uuid4())
    plan = RoutinePlan.model_validate(
        {
            "schema_version": "1.0",
            "routine_type": "HIPERTROFIA",
            "target_weekly_frequency": 3,
            "explanation": "Candidata con foco en tríceps.",
            "days": [
                {
                    "position": day,
                    "name": f"Día {day}",
                    "dominant_pattern": "AISLAMIENTO_SUPERIOR",
                    "exercises": [
                        {
                            "position": position,
                            "exercise_id": exercise_id,
                            "set_groups": [
                                {
                                    "count": 3,
                                    "min_repetitions": 8,
                                    "max_repetitions": 12,
                                    "rest_seconds": 90,
                                }
                            ],
                        }
                        for position, exercise_id in enumerate([*triceps_ids, press_id], start=1)
                    ],
                }
                for day in range(1, 4)
            ],
        }
    )
    preferences = {
        "muscle_counts_per_day": [{"muscle": "TRICEPS", "count": 3}],
        "allowed_catalog": [
            {
                "id": exercise_id,
                "movement_pattern": "AISLAMIENTO_SUPERIOR",
                "primary_muscles": ["TRICEPS"],
            }
            for exercise_id in triceps_ids
        ]
        + [
            {
                "id": press_id,
                "movement_pattern": "EMPUJE_HORIZONTAL",
                "primary_muscles": ["PECTORAL"],
                "secondary_muscles": ["TRICEPS"],
            }
        ],
    }
    plan.validate_prescription({}, preferences)
    plan.days[1].exercises.pop(2)
    with raises(InvalidPrescriptionError, match="requested_muscle_count"):
        plan.validate_prescription({}, preferences)
    plan.days[1].exercises.insert(2, plan.days[1].exercises[0])
    with raises(InvalidPrescriptionError, match="requested_muscle_count"):
        plan.validate_prescription({}, preferences)


def test_ollama_rejects_a_truncated_stream(monkeypatch: MonkeyPatch) -> None:
    original_async_client = httpx.AsyncClient

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"message":{"content":"{}"},"done":false}\n')

    def mock_async_client(**kwargs: object) -> httpx.AsyncClient:
        return original_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", mock_async_client)
    client = OllamaClient("https://polo.example.invalid/polo", "qwen3.5:9b", "token", 120)
    with raises(ValueError, match="before generation completed"):
        asyncio.run(client.generate_routine({}, {"allowed_catalog": [{"id": str(uuid4())}]}))


def test_ollama_rejects_a_missing_catalog_before_inference() -> None:
    client = OllamaClient("https://polo.example.invalid/polo", "qwen3.5:9b", "token", 120)
    with raises(InvalidPrescriptionError, match="No allowed catalog"):
        asyncio.run(client.generate_routine({}, {}))


def test_ollama_readiness_checks_installed_model(
    monkeypatch: MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"models": [{"name": "another-model:latest"}]})

    original_async_client = httpx.AsyncClient

    def mock_async_client(**kwargs: object) -> httpx.AsyncClient:
        return original_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", mock_async_client)
    client = OllamaClient("https://polo.example.invalid/polo", "qwen3.5:9b", "token", 120)
    with raises(RuntimeError, match="model is not installed"):
        asyncio.run(client.ping())
    assert seen[0].headers["authorization"] == "Bearer token"


def test_prescription_constraints_reject_invalid_model_choices_without_changing_them() -> None:
    exercise_id = str(uuid4())
    plan = RoutinePlan.model_validate(
        {
            "schema_version": "1.0",
            "routine_type": "HIPERTROFIA",
            "target_weekly_frequency": 3,
            "explanation": "Candidata.",
            "days": [
                {
                    "position": day + 1,
                    "name": f"Día {day + 1}",
                    "dominant_pattern": "CORE",
                    "exercises": [
                        {
                            "position": index + 1,
                            "exercise_id": exercise_id,
                            "set_groups": [
                                {
                                    "count": 3,
                                    "min_repetitions": 8,
                                    "max_repetitions": 12,
                                    "rest_seconds": 90,
                                }
                            ],
                        }
                        for index in range(5)
                    ],
                }
                for day in range(3)
            ],
        }
    )
    preferences = {
        "allowed_catalog": [{"id": exercise_id, "movement_pattern": "CORE"}],
        "prescription_constraints": {
            "purposes": {
                "HIPERTROFIA": {
                    "weekly_frequency": [3, 6],
                    "exercises_per_day": [5, 8],
                    "work_sets_per_exercise": [3, 4],
                    "repetitions": [6, 12],
                    "rest_seconds": [60, 120],
                }
            },
            "required_pattern_groups": [["CORE"]],
        },
    }
    context = {"available_days_per_week": 3}
    plan.validate_prescription(context, preferences)
    group = plan.days[0].exercises[0].set_groups[0]
    group.count = 2
    group.max_repetitions = 20
    group.rest_seconds = 30
    with raises(InvalidPrescriptionError) as rejected:
        plan.validate_prescription(context, preferences)
    assert "work_sets_per_exercise" in str(rejected.value)
    assert "repetitions" in str(rejected.value)
    assert "rest_seconds" in str(rejected.value)
    assert group.count == 2
    assert group.max_repetitions == 20
    assert group.rest_seconds == 30
