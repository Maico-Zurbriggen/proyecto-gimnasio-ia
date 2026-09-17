"""Fixtures compartidas.

No hay fixture de Postgres real: la persistencia usa jsonb, enums de Postgres y
FOR UPDATE SKIP LOCKED, que sqlite no soporta. Los tests de la API/auth usan
FakeRepository (este archivo) en vez de golpear una base real -- documentado,
no hay integración contra Postgres real en este entorno.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from gym_engine.persistence.models import RequestStatus


class FakeConnection:
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeRepository:
    """Reemplaza gym_engine.persistence.repository en tests de la API/auth."""

    def __init__(self) -> None:
        self._requests: dict[uuid.UUID, dict[str, Any]] = {}

    def get_by_idempotency_key(self, conn: object, idempotency_key: str) -> dict[str, Any] | None:
        for request_id, row in self._requests.items():
            if row["idempotency_key"] == idempotency_key:
                return {"id": request_id, "state": row["state"]}
        return None

    def create_request(
        self,
        conn: object,
        *,
        idempotency_key: str,
        minimized_context: dict[str, Any],
        preferences: dict[str, Any],
        retention_days: int,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4()
        self._requests[request_id] = {
            "idempotency_key": idempotency_key,
            "state": "PENDIENTE",
            "minimized_context": minimized_context,
            "preferences": preferences,
            "structured_output": None,
            "error_code": None,
        }
        return {"id": request_id, "state": "PENDIENTE"}

    def get_status(self, conn: object, request_id: uuid.UUID) -> RequestStatus | None:
        row = self._requests.get(request_id)
        if row is None:
            return None
        return RequestStatus(
            request_id=request_id,
            state=row["state"],
            structured_output=row["structured_output"],
            error_code=row["error_code"],
        )

    def set_state(
        self,
        request_id: uuid.UUID,
        *,
        state: str,
        structured_output: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        self._requests[request_id].update(
            state=state, structured_output=structured_output, error_code=error_code
        )


@pytest.fixture
def fake_repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def fake_connection() -> FakeConnection:
    return FakeConnection()
