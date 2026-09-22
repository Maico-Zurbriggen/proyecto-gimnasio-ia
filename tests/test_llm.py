from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx
from pytest import MonkeyPatch

from gym_engine.llm import OllamaClient


def test_ping_sends_llm_token_as_bearer(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = OllamaClient(
        base_url="https://llm.example.invalid",
        model="qwen2.5:7b-instruct",
        api_token="llm-secret",
        timeout_seconds=120,
    )
    asyncio.run(client.ping())

    assert captured["headers"] == {"Authorization": "Bearer llm-secret"}
    assert "auth" not in captured


def valid_routine_output() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "routine_type": "HIPERTROFIA",
        "target_weekly_frequency": 3,
        "days": [
            {
                "position": 1,
                "name": "Día 1",
                "dominant_pattern": "DOMINANTE_RODILLA",
                "exercises": [
                    {
                        "position": 1,
                        "exercise_id": str(uuid4()),
                        "sets": [
                            {
                                "position": 1,
                                "min_repetitions": 8,
                                "max_repetitions": 12,
                                "rest_seconds": 90,
                            }
                        ],
                    }
                ],
            }
        ],
        "explanation": "Rutina de prueba.",
    }


def find_keys(node: Any, wanted: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in wanted:
                found.add(key)
            found |= find_keys(value, wanted)
    elif isinstance(node, list):
        for value in node:
            found |= find_keys(value, wanted)
    return found


def test_generate_routine_sends_format_schema_without_refs(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    content = json.dumps(valid_routine_output(), ensure_ascii=False)
    half = len(content) // 2
    lines = [
        json.dumps(
            {
                "model": "qwen3.5:9b",
                "message": {"role": "assistant", "thinking": "pensando..."},
                "done": False,
            }
        ),
        json.dumps(
            {
                "model": "qwen3.5:9b",
                "message": {"role": "assistant", "content": content[:half]},
                "done": False,
            }
        ),
        json.dumps(
            {
                "model": "qwen3.5:9b",
                "message": {"role": "assistant", "content": content[half:]},
                "done": False,
            }
        ),
        json.dumps({"model": "qwen3.5:9b", "done": True, "done_reason": "stop"}),
    ]

    class FakeStream:
        async def __aenter__(self) -> FakeStream:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> Any:
            for line in lines:
                yield line

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, _method: str, _url: str, json: Any) -> FakeStream:
            captured["payload"] = json
            return FakeStream()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = OllamaClient(
        base_url="https://llm.example.invalid",
        model="qwen3.5:9b",
        api_token="llm-secret",
        timeout_seconds=120,
    )
    result = asyncio.run(
        client.generate_routine({"nivel": "principiante"}, {"texto": "x"})
    )

    assert captured["payload"]["stream"] is True
    assert result.output.routine_type == "HIPERTROFIA"
    assert result.model_version == "qwen3.5:9b"
    assert len(result.output_hash) == 64

    fmt = captured["payload"]["format"]
    assert find_keys(fmt, {"$ref", "$defs", "minLength", "maxLength"}) == set()
    assert set(fmt["properties"]) == {
        "schema_version",
        "routine_type",
        "target_weekly_frequency",
        "days",
        "uncovered_patterns",
        "explanation",
    }
    day_props = fmt["properties"]["days"]["items"]["properties"]
    assert "CORE" in day_props["dominant_pattern"]["enum"]
