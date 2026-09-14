import httpx
import pytest

from gym_engine.config import Settings
from gym_engine.llm.client import (
    LlmInvalidOutput,
    LlmUnavailable,
    OllamaClient,
    OpenAiClient,
    build_llm_client,
)
from gym_engine.llm.schemas import ParametrosRutina


def _settings() -> Settings:
    return Settings(
        database_url="sqlite://", llm_api_url="http://ollama.local", ollama_model="test-model"
    )


def test_generate_structured_success(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ParametrosRutina(
        objetivo="fuerza", frecuencia_semanal=4, duracion_minutos=50
    ).model_dump_json()

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"response": body}, request=httpx.Request("POST", url))

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OllamaClient(_settings())
    result = client.generate_structured("prompt", ParametrosRutina)

    assert result.objetivo == "fuerza"
    assert result.frecuencia_semanal == 4


def test_generate_structured_timeout_raises_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("timeout", request=httpx.Request("POST", url))

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OllamaClient(_settings())
    with pytest.raises(LlmUnavailable):
        client.generate_structured("prompt", ParametrosRutina)


def test_generate_structured_invalid_json_raises_llm_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200, json={"response": "{not valid}"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OllamaClient(_settings())
    with pytest.raises(LlmInvalidOutput):
        client.generate_structured("prompt", ParametrosRutina)


def _openai_settings() -> Settings:
    return Settings(
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
            json={"choices": [{"message": {"content": body}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("gym_engine.llm.client.httpx.post", fake_post)

    client = OpenAiClient(_openai_settings())
    result = client.generate_structured("prompt", ParametrosRutina)

    assert result.objetivo == "fuerza"


def test_openai_without_api_key_raises_llm_unavailable() -> None:
    settings = Settings(database_url="sqlite://", llm_provider="openai", openai_api_key=None)
    with pytest.raises(LlmUnavailable):
        OpenAiClient(settings)


def test_build_llm_client_selects_provider() -> None:
    assert isinstance(build_llm_client(_settings()), OllamaClient)
    assert isinstance(build_llm_client(_openai_settings()), OpenAiClient)


def test_build_llm_client_rejects_unknown_provider() -> None:
    settings = Settings(database_url="sqlite://", llm_provider="mistral")
    with pytest.raises(LlmUnavailable):
        build_llm_client(settings)
