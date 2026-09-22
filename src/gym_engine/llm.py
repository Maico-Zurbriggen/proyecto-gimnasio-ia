from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

from .contracts import RoutineGenerationOutput


@dataclass(frozen=True)
class LlmResult:
    output: RoutineGenerationOutput
    output_hash: str
    model_version: str


# Keywords rejected by Ollama's grammar builder (HTTP 400 "failed to parse
# grammar"). Length limits stay enforced by validating the model response
# against the full contract, so dropping them from `format` only relaxes
# the generation guide, not the accepted output.
OLLAMA_UNSUPPORTED_SCHEMA_KEYS = frozenset({"minLength", "maxLength"})


def ollama_format_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return the model JSON schema adapted to Ollama's `format` field.

    Local ``$defs`` references are inlined and keywords unsupported by
    Ollama's constrained-decoding grammar builder are removed.
    """
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})
    inlined = _inline_refs(schema, defs)
    return _strip_keys(inlined, OLLAMA_UNSUPPORTED_SCHEMA_KEYS)


def _strip_keys(node: Any, drop: frozenset[str]) -> Any:
    if isinstance(node, dict):
        return {k: _strip_keys(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip_keys(v, drop) for v in node]
    return node


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if set(node) == {"$ref"}:
            target = node["$ref"]
            if isinstance(target, str) and target.startswith("#/$defs/"):
                name = target.rsplit("/", 1)[1]
                return _inline_refs(copy.deepcopy(defs[name]), defs)
            return dict(node)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_token: str,
        timeout_seconds: int,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self._timeout = timeout_seconds

    async def ping(self) -> None:
        async with httpx.AsyncClient(
            headers=self._headers, timeout=10, follow_redirects=False
        ) as client:
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()

    async def generate_routine(
        self,
        minimized_context: dict[str, Any],
        preferences: dict[str, Any],
    ) -> LlmResult:
        payload = {
            "model": self._model,
            "stream": True,
            "format": ollama_format_schema(RoutineGenerationOutput),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Sos un asistente de prescripción de entrenamiento. Respondé sólo "
                        "con JSON válido para el schema recibido. Usá exclusivamente ids de "
                        "ejercicios presentes en el catálogo permitido. No inventes datos, no "
                        "emitas diagnósticos ni consejos médicos."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "generarRutina",
                            "minimized_context": minimized_context,
                            "preferences": preferences,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "options": {"temperature": 0.2},
        }
        # Streamed: full generations take minutes and a single buffered
        # response trips the tunnel/proxy idle timeout (HTTP 524), while a
        # stream of chunks keeps the connection alive.
        try:
            async with asyncio.timeout(self._timeout):
                async with httpx.AsyncClient(
                    headers=self._headers,
                    timeout=httpx.Timeout(self._timeout),
                    follow_redirects=False,
                ) as client:
                    async with client.stream(
                        "POST", f"{self._base_url}/api/chat", json=payload
                    ) as response:
                        response.raise_for_status()
                        content_parts: list[str] = []
                        model_version = self._model
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            message = chunk.get("message", {})
                            content_parts.append(message.get("content", ""))
                            if chunk.get("model"):
                                model_version = str(chunk["model"])
        except TimeoutError as error:
            raise httpx.TimeoutException(
                f"LLM streaming timed out after {self._timeout}s"
            ) from error

        raw_content = "".join(content_parts)
        parsed = RoutineGenerationOutput.model_validate_json(raw_content)
        canonical = json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return LlmResult(
            output=parsed,
            output_hash=hashlib.sha256(canonical).hexdigest(),
            model_version=model_version,
        )
