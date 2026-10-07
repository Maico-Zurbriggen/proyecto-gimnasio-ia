from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx
from pytest import MonkeyPatch, mark, raises

from gym_engine.llm import OllamaClient
from gym_engine.llm_v2 import ContextCapacityError, Plan


def proposal(index: int = 0) -> dict[str, Any]:
    return {
        "outcome": "PROPOSED",
        "routine_type": "FUERZA",
        "explanation": "Revisar la adaptación.",
        "warnings": ["Requiere evaluación del entrenador."],
        "missing_information": [],
        "days": [
            {
                "name": "Día",
                "dominant_pattern": "CORE",
                "exercises": [
                    {
                        "index": index,
                        "note": None,
                        "groups": [
                            {
                                "count": 1,
                                "min_repetitions": 1,
                                "max_repetitions": 2,
                                "rest_seconds": 1000,
                                "suggested_load": None,
                                "warmup": False,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def transport(
    monkeypatch: MonkeyPatch,
    value: dict[str, Any],
    *,
    context_limit: int = 262144,
    done: bool = True,
    done_reason: str = "stop",
    allocated_context: int = 262144,
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/api/show"):
            return httpx.Response(200, json={"model_info": {"qwen.context_length": context_limit}})
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(
                200, json={"models": [{"name": "test-model:latest", "digest": "test-revision"}]}
            )
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "test-model:latest",
                            "digest": "test-revision",
                            "context_length": allocated_context,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "done": done,
                    "done_reason": done_reason,
                    "prompt_eval_count": 5000,
                    "model": "test-model",
                    "message": {"content": json.dumps(value)},
                }
            ),
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handle)),
    )
    return seen


def client(context_tokens: int = 131072) -> OllamaClient:
    return OllamaClient(
        base_url="https://llm.test/polo",
        model="test-model",
        api_token="test-only",
        timeout_seconds=5,
        context_tokens=context_tokens,
        response_tokens=4096,
    )


def catalog(count: int = 80) -> list[dict[str, Any]]:
    return [
        {
            "id": str(uuid4()),
            "name": f"Synthetic exercise {index}",
            "movement_pattern": "CORE",
            "instructions": "Instrucciones completas de prueba.",
            "primary_muscles": ["CORE", "OTRO"],
            "secondary_muscles": [],
            "difficulty_level": "AVANZADO",
            "equipment": ["MANCUERNAS"],
            "joints": ["HOMBRO"],
            "unilateral": True,
            "revision": 2,
            "availability_revision": 4,
        }
        for index in range(count)
    ]


def test_all_exercises_and_raw_preferences_reach_model_without_training_filters(
    monkeypatch: MonkeyPatch,
) -> None:
    entries = catalog()
    seen = transport(monkeypatch, proposal(79))
    result = asyncio.run(
        client().generate_routine(
            {
                "experience_level": "PRINCIPIANTE",
                "available_days_per_week": 1,
                "conditions": [{"body_zone": "HOMBRO", "severity": "SEVERA"}],
                "history": {
                    "exercise_performance": [
                        {"exercise_id": entries[79]["id"], "average_load": 20}
                    ],
                    "current_routine": {"exercise_ids": [entries[79]["id"], "unavailable-history"]},
                },
            },
            {
                "schema_version": "2.0",
                "allowed_catalog": entries,
                "free_text": "3 ejercicios de tríceps por día",
                "parameters": None,
            },
        )
    )
    chat = next(request for request in seen if request.url.path.endswith("/api/chat"))
    payload = json.loads(chat.content)
    data = json.loads(payload["messages"][1]["content"])
    assert len(data["preferences"]["allowed_catalog"]) == 80
    assert data["preferences"]["allowed_catalog"][-1]["instructions"] == entries[-1]["instructions"]
    assert data["preferences"]["free_text"] == "3 ejercicios de tríceps por día"
    assert "prescription_constraints" not in data["preferences"]
    assert "muscle_counts_per_day" not in data["preferences"]
    assert data["student_context"]["history"]["exercise_performance"][0]["exercise_reference"] == {
        "index": 79
    }
    assert data["student_context"]["history"]["current_routine"]["exercise_references"][1] == {
        "id": "unavailable-history",
        "outside_catalog": True,
    }
    assert (
        str(result.output.model_dump()["days"][0]["exercises"][0]["exercise_id"])
        == entries[79]["id"]
    )
    assert result.output.model_dump()["days"][0]["dominant_pattern"] == "CORE"
    assert payload["options"]["num_ctx"] == 131072
    assert chat.headers["Authorization"] == "Bearer test-only"
    assert result.model_version == "test-model@test-revision"


def test_unable_is_an_explicit_model_decision_without_a_fallback(monkeypatch: MonkeyPatch) -> None:
    value = {
        "outcome": "UNABLE",
        "routine_type": None,
        "days": [],
        "explanation": "Falta información esencial.",
        "warnings": [],
        "missing_information": ["Indicaciones del profesional."],
    }
    transport(monkeypatch, value)
    result = asyncio.run(
        client().generate_routine({}, {"schema_version": "2.0", "allowed_catalog": catalog(1)})
    )
    assert result.output.model_dump()["outcome"] == "UNABLE"
    assert "days" not in result.output.model_dump()


def test_full_context_capacity_failure_happens_before_network(monkeypatch: MonkeyPatch) -> None:
    seen = transport(monkeypatch, proposal())
    with raises(ContextCapacityError):
        asyncio.run(
            client(4096).generate_routine(
                {}, {"schema_version": "2.0", "allowed_catalog": catalog()}
            )
        )
    assert seen == []


def test_model_window_must_support_configured_capacity(monkeypatch: MonkeyPatch) -> None:
    seen = transport(monkeypatch, proposal(), context_limit=8192)
    with raises(ContextCapacityError):
        asyncio.run(
            client().generate_routine({}, {"schema_version": "2.0", "allowed_catalog": catalog(1)})
        )
    assert len(seen) == 1


@mark.parametrize("done,reason", [(False, "stop"), (True, "length")])
def test_incomplete_output_is_rejected(monkeypatch: MonkeyPatch, done: bool, reason: str) -> None:
    transport(monkeypatch, proposal(), done=done, done_reason=reason)
    with raises(ValueError):
        asyncio.run(
            client().generate_routine({}, {"schema_version": "2.0", "allowed_catalog": catalog(1)})
        )


def test_snapshot_indices_and_storage_intervals_are_checked() -> None:
    with raises(ValueError, match="outside exact snapshot"):
        Plan.model_validate(proposal(1)).expand(catalog(1))
    value = proposal()
    value["days"][0]["exercises"][0]["groups"][0]["max_repetitions"] = 0
    with raises(ValueError):
        Plan.model_validate(value).expand(catalog(1))


def test_expansion_enforces_capacity_across_all_exercises_before_allocating() -> None:
    value = proposal()
    exercise = value["days"][0]["exercises"][0]
    exercise["groups"][0]["count"] = 500
    value["days"][0]["exercises"].append(
        {**exercise, "groups": [{**exercise["groups"][0], "count": 1}]}
    )
    with raises(ValueError, match="transport capacity"):
        Plan.model_validate(value).expand(catalog(1))


def test_server_must_actually_allocate_the_requested_window(monkeypatch: MonkeyPatch) -> None:
    transport(monkeypatch, proposal(), allocated_context=8192)
    with raises(ContextCapacityError, match="runtime_context_capacity_unsupported"):
        asyncio.run(
            client().generate_routine({}, {"schema_version": "2.0", "allowed_catalog": catalog(1)})
        )
