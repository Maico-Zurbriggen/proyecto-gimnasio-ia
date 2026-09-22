from __future__ import annotations

import copy
import json
from typing import Any, Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from gym_engine.config import Settings

ModelT = TypeVar("ModelT", bound=BaseModel)

OLLAMA_UNSUPPORTED_SCHEMA_KEYS = frozenset({"minLength", "maxLength"})


def _inline_schema_refs(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if set(node) == {"$ref"}:
            target = node["$ref"]
            if isinstance(target, str) and target.startswith("#/$defs/"):
                name = target.rsplit("/", 1)[1]
                return _inline_schema_refs(copy.deepcopy(definitions[name]), definitions)
            return dict(node)
        return {
            key: _inline_schema_refs(value, definitions)
            for key, value in node.items()
            if key != "$defs"
        }
    if isinstance(node, list):
        return [_inline_schema_refs(value, definitions) for value in node]
    return node


def _strip_schema_keys(node: Any, keys: frozenset[str]) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_schema_keys(value, keys)
            for key, value in node.items()
            if key not in keys
        }
    if isinstance(node, list):
        return [_strip_schema_keys(value, keys) for value in node]
    return node


def ollama_format_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Adapta JSON Schema al subconjunto aceptado por Ollama.

    La validacion final sigue usando el modelo Pydantic completo; relajar la
    guia de generacion no relaja el contrato que acepta el servicio.
    """
    raw = schema.model_json_schema()
    inlined = _inline_schema_refs(raw, raw.get("$defs", {}))
    return cast(
        dict[str, Any],
        _strip_schema_keys(inlined, OLLAMA_UNSUPPORTED_SCHEMA_KEYS),
    )


class LlmUnavailable(Exception):
    """El LLM no respondió (timeout, conexión rechazada, error de servidor)."""


class LlmInvalidOutput(Exception):
    """El LLM respondió pero la salida no cumple el esquema pedido."""


class LlmClient(Protocol):
    """Conector privado hacia el LLM. Único punto por el que se lo llama (AGENTS.md)."""

    def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT: ...


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.llm_api_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.generation_timeout_seconds
        self._api_key = settings.llm_api_token

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "format": ollama_format_schema(schema),
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                chunks: list[str] = []
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    value = chunk.get("response", "")
                    if isinstance(value, str):
                        chunks.append(value)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            raise LlmUnavailable(str(exc)) from exc
        except (json.JSONDecodeError, AttributeError) as exc:
            raise LlmInvalidOutput(str(exc)) from exc

        raw = "".join(chunks)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LlmInvalidOutput(str(exc)) from exc


class OpenAiClient:
    """Conector privado hacia la Responses API de OpenAI (structured outputs)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LlmUnavailable("OPENAI_API_KEY no configurada")
        self._base_url = settings.openai_base_url.rstrip("/")
        self._model = settings.openai_model
        self._timeout = settings.generation_timeout_seconds
        self._api_key = settings.openai_api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        payload = {
            "model": self._model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            },
        }
        try:
            response = httpx.post(
                f"{self._base_url}/responses",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            raise LlmUnavailable(str(exc)) from exc

        raw = self._extract_output_text(response.json())
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LlmInvalidOutput(str(exc)) from exc

    @staticmethod
    def _extract_output_text(body: dict[str, Any]) -> str:
        output = body.get("output", [])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            return text
        raise LlmInvalidOutput(f"respuesta de OpenAI sin output_text: {body}")


def build_llm_client(settings: Settings) -> LlmClient:
    if settings.llm_provider == "openai":
        return OpenAiClient(settings)
    if settings.llm_provider == "ollama":
        return OllamaClient(settings)
    raise LlmUnavailable(f"LLM_PROVIDER desconocido: {settings.llm_provider!r}")
