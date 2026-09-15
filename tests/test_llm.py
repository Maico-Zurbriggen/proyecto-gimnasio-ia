from __future__ import annotations

import asyncio
from typing import Any

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
