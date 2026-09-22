import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader

from gym_engine.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _bearer_token(authorization: str | None) -> str | None:
    scheme, _, token = (authorization or "").partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def verify_api_key(
    api_key: Annotated[str | None, Depends(_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.expected_api_key
    if not expected:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "AI_SERVICE_API_KEY no configurada del lado del servicio",
        )
    # Backend histórico manda `X-API-Key`; el flujo heredado de Vercel Queues manda
    # `Authorization: Bearer`. Se aceptan ambos esquemas mientras conviven los dos clientes.
    candidate = api_key or _bearer_token(authorization)
    if not candidate or not secrets.compare_digest(candidate, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key invalida o ausente")


RequireApiKey = Depends(verify_api_key)
