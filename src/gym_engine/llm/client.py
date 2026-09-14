from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from gym_engine.config import Settings

ModelT = TypeVar("ModelT", bound=BaseModel)


class LlmUnavailable(Exception):
    """El LLM no respondió (timeout, conexión rechazada, error de servidor)."""


class LlmInvalidOutput(Exception):
    """El LLM respondió pero la salida no cumple el esquema pedido."""


class OllamaClient:
    """Conector privado hacia Ollama. Único punto por el que se llama al LLM (AGENTS.md)."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.llm_api_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.generation_timeout_seconds
        self._api_key = settings.llm_api_key

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "format": schema.model_json_schema(),
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            raise LlmUnavailable(str(exc)) from exc

        raw = response.json().get("response", "")
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LlmInvalidOutput(str(exc)) from exc
