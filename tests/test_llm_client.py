from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gym_engine.config import Settings
from gym_engine.llm.client import (
    LlmInvalidOutput,
    LlmUnavailable,
    OllamaClient,
    OpenAiClient,
    build_llm_client,
    ollama_format_schema,
)
from gym_engine.llm.schemas import ParametrosRutina, RutinaEstructurada


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        llm_provider="ollama",
        llm_api_url="http://ollama.local",
        ollama_model="test-model",
    )


def test_generate_structured_success(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ParametrosRutina(
        objetivo="fuerza", frecuencia_semanal=4, duracion_minutos=50
    ).model_dump_json()

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            half = len(body) // 2
            return [
                json.dumps({"response": body[:half], "done": False}),
                json.dumps({"response": body[half:], "done": True}),
            ]

    def fake_stream(*_args: object, **_kwargs: object) -> FakeStream:
        return FakeStream()

    monkeypatch.setattr("gym_engine.llm.client.httpx.stream", fake_stream)

    client = OllamaClient(_settings())
    result = client.generate_structured("prompt", ParametrosRutina)

    assert result.objetivo == "fuerza"
    assert result.frecuencia_semanal == 4


def test_generate_structured_timeout_raises_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(*_args: object, **_kwargs: object) -> Any:
        raise httpx.TimeoutException("timeout", request=httpx.Request("POST", url))

    url = "http://ollama.local/api/generate"
    monkeypatch.setattr("gym_engine.llm.client.httpx.stream", fake_stream)

    client = OllamaClient(_settings())
    with pytest.raises(LlmUnavailable):
        client.generate_structured("prompt", ParametrosRutina)


def test_generate_structured_invalid_json_raises_llm_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return [json.dumps({"response": "{not valid}", "done": True})]

    monkeypatch.setattr(
        "gym_engine.llm.client.httpx.stream",
        lambda *_args, **_kwargs: FakeStream(),
    )

    client = OllamaClient(_settings())
    with pytest.raises(LlmInvalidOutput):
        client.generate_structured("prompt", ParametrosRutina)


def test_ollama_format_schema_inlines_refs_and_drops_unsupported_keys() -> None:
    formatted = ollama_format_schema(RutinaEstructurada)

    def find_keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node).union(*(find_keys(value) for value in node.values()))
        if isinstance(node, list):
            return set().union(*(find_keys(value) for value in node))
        return set()

    assert find_keys(formatted).isdisjoint(
        {"$ref", "$defs", "minLength", "maxLength"}
    )


def _openai_settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        llm_provider="openai",
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
    )


def test_openai_generate_structured_success(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ParametrosRutina(
        objetivo="fuerza", frecuencia_semanal=4, duracion_minutos=50
    ).model_dump_json()

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": body}],
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OpenAiClient(_openai_settings())
    result = client.generate_structured("prompt", ParametrosRutina)

    assert result.objetivo == "fuerza"


def test_openai_missing_output_text_raises_llm_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"output": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OpenAiClient(_openai_settings())
    with pytest.raises(LlmInvalidOutput):
        client.generate_structured("prompt", ParametrosRutina)


def test_openai_without_api_key_raises_llm_unavailable() -> None:
    settings = Settings(
        _env_file=None, database_url="sqlite://", llm_provider="openai", openai_api_key=None
    )
    with pytest.raises(LlmUnavailable):
        OpenAiClient(settings)


def test_build_llm_client_selects_provider() -> None:
    assert isinstance(build_llm_client(_settings()), OllamaClient)
    assert isinstance(build_llm_client(_openai_settings()), OpenAiClient)


def test_build_llm_client_rejects_unknown_provider() -> None:
    settings = Settings(_env_file=None, database_url="sqlite://", llm_provider="mistral")
    with pytest.raises(LlmUnavailable):
        build_llm_client(settings)
