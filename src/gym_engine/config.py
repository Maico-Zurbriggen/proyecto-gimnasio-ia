from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()


class ConfigurationError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_url: str
    api_key: str
    generation_queue_mode: Literal["vercel", "local"]
    queue_region: str
    llm_api_url: str
    llm_model: str
    llm_api_token: str
    configuration_version: str
    timeout_seconds: int
    max_attempts: int

    @classmethod
    def from_env(cls) -> Settings:
        retries = int(os.getenv("GENERATION_MAX_RETRIES", "1"))
        if retries != 1:
            raise ConfigurationError("GENERATION_MAX_RETRIES must be 1")
        app_env = os.getenv("APP_ENV", "test").strip().lower()
        queue_mode = os.getenv("GENERATION_QUEUE_MODE", "vercel").strip().lower()
        if queue_mode == "vercel":
            generation_queue_mode: Literal["vercel", "local"] = "vercel"
        elif queue_mode == "local":
            generation_queue_mode = "local"
        else:
            raise ConfigurationError(
                "GENERATION_QUEUE_MODE must be either 'vercel' or 'local'"
            )
        if generation_queue_mode == "local" and app_env != "local":
            raise ConfigurationError(
                "GENERATION_QUEUE_MODE=local is only allowed with APP_ENV=local"
            )
        if app_env == "production" and generation_queue_mode != "vercel":
            raise ConfigurationError(
                "Production must use the durable Vercel generation queue"
            )
        database_url = _required("DATABASE_URL")
        if app_env == "local" and urlsplit(database_url).hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ConfigurationError("APP_ENV=local requires a local PostgreSQL database")
        return cls(
            app_env=app_env,
            database_url=database_url,
            api_key=_required("AI_SERVICE_API_KEY"),
            generation_queue_mode=generation_queue_mode,
            queue_region=os.getenv("QUEUE_REGION", "gru1").strip(),
            llm_api_url=_required("LLM_API_URL").rstrip("/"),
            llm_model=_required("LLM_MODEL"),
            llm_api_token=_required("LLM_API_TOKEN"),
            configuration_version=os.getenv(
                "LLM_CONFIGURATION_VERSION", "generative/generar-rutina@10"
            ).strip(),
            timeout_seconds=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "120")),
            max_attempts=retries + 1,
        )
