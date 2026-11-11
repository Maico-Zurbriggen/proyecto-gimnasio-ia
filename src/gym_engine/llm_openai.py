from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
MAX_RESPONSE_BYTES = 2_000_000


def openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def openai_ping(base_url: str, model: str, headers: dict[str, str]) -> None:
    async with httpx.AsyncClient(headers=headers, timeout=10, follow_redirects=False) as client:
        response = await client.get(f"{base_url}/models/{model}")
        response.raise_for_status()


async def openai_chat_json(
    *,
    base_url: str,
    model: str,
    headers: dict[str, str],
    timeout: int,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    response_tokens: int,
) -> tuple[str, str]:
    """Run one Responses API call constrained to `schema`; return (content, model)."""
    payload = {
        "model": model,
        "input": messages,
        "max_output_tokens": response_tokens,
        "store": False,
        "text": {
            "format": {"type": "json_schema", "name": "routine", "schema": schema, "strict": False}
        },
    }
    async with (
        asyncio.timeout(timeout),
        httpx.AsyncClient(
            headers=headers, timeout=httpx.Timeout(timeout, connect=10), follow_redirects=False
        ) as client,
    ):
        response = await client.post(f"{base_url}/responses", json=payload)
        if response.is_error:
            detail = (response.json().get("error") or {}) if response.content else {}
            raise ValueError(
                f"OpenAI HTTP {response.status_code}: {detail.get('code')} "
                f"param={detail.get('param')} {str(detail.get('message'))[:80]}"
            )
        body = response.json()
    if body.get("error"):
        raise ValueError("LLM reported an inference error")
    if body.get("status") == "incomplete":
        raise ValueError("LLM output exhausted its token budget")
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if block.get("type") == "refusal":
                raise ValueError("LLM refused the request")
            if block.get("type") == "output_text":
                parts.append(block.get("text", ""))
    content = "".join(parts)
    if len(content.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("LLM response exceeded byte capacity")
    if not content:
        raise ValueError("LLM returned empty content")
    json.loads(content)
    return content, str(body.get("model", model))
