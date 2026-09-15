from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

from .contracts import RoutineGenerationOutput


@dataclass(frozen=True)
class LlmResult:
    output: RoutineGenerationOutput
    output_hash: str
    model_version: str


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
            "stream": False,
            "format": RoutineGenerationOutput.model_json_schema(),
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
        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=False,
        ) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            envelope = response.json()

        raw_content = envelope["message"]["content"]
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
            model_version=str(envelope.get("model", self._model)),
        )
