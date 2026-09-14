"""Mapeo declarativo de las tablas ai_generation_* (Neon, rol restringido del servicio IA).

Este repositorio no ejecuta migraciones (README.md): las tablas ya existen. Los nombres de
columna se infieren de la narrativa de maquina-estados-langgraph.pdf (§3 y §11) a falta del
data-interface.md real del repositorio documental — provisorio hasta confirmarse contra el
esquema real.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AiGenerationRequest(Base):
    __tablename__ = "ai_generation_requests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True, index=True)
    gym_id: Mapped[uuid.UUID] = mapped_column(index=True)
    student_id: Mapped[uuid.UUID] = mapped_column(index=True)
    requested_by_user_id: Mapped[uuid.UUID]
    texto_libre: Mapped[str | None] = mapped_column(String, default=None)
    parametros: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    catalogo_prefiltrado: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    contexto_minimizado: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class AiGenerationAttempt(Base):
    __tablename__ = "ai_generation_attempts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ai_generation_requests.id"), index=True
    )
    node: Mapped[str] = mapped_column(String)
    attempt_number: Mapped[int]
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(String, default=None)


class AiGenerationResult(Base):
    __tablename__ = "ai_generation_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ai_generation_requests.id"), index=True
    )
    estructura_candidata: Mapped[dict[str, Any]] = mapped_column(JSON)
    version_modelo: Mapped[str] = mapped_column(String)
    version_prompt: Mapped[str] = mapped_column(String)
    ruta: Mapped[str] = mapped_column(String, default="GENERATIVA")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AiResultValidation(Base):
    __tablename__ = "ai_result_validations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ai_generation_requests.id"), index=True
    )
    valid: Mapped[bool]
    violations: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
