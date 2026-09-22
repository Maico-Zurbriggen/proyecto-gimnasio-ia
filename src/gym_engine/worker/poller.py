"""Entrypoint local/self-hosted para el worker (AGENTS.md).

En Vercel, gym_engine.worker.queue_consumer corre como push consumer generado a partir de
[[tool.vercel.subscribers]] en pyproject.toml -- este script no se usa ahi. Fuera de Vercel
(desarrollo local u otro proceso self-hosted) hace falta algo que efectivamente pida trabajo
a Vercel Queues: este modulo corre el mismo handler (handle_routine_generation) en poll mode
via el SDK, en vez del SQL polling directo contra Postgres que tenia la version anterior.

Requiere que el proyecto este vinculado a Vercel (`vercel link`) para que el SDK resuelva un
token OIDC al pollear, o bien VERCEL_QUEUE_TOKEN seteado a mano.
"""

from __future__ import annotations

import logging

from vercel.queue.sync import QueueClient

from gym_engine.config import get_settings
from gym_engine.worker.queue_consumer import handle_routine_generation

logger = logging.getLogger(__name__)


def run_forever() -> None:
    settings = get_settings()
    logger.info("Poller (poll mode) iniciado, intervalo=%ss", settings.poller_interval_seconds)
    queue = QueueClient(region=settings.queue_region)
    # limit=1: misma concurrencia=1 que el push consumer (AGENTS.md), para no pegarle a
    # Ollama con mas de un intento en simultaneo.
    future = queue.poll_and_handle(
        handle_routine_generation,
        interval=settings.poller_interval_seconds,
        limit=1,
    )
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
