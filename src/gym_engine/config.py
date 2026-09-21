from __future__ import annotations

import os
from dataclasses import dataclass

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
        return cls(
            app_env=os.getenv("APP_ENV", "test").strip(),
            database_url=_required("DATABASE_URL"),
            api_key=_required("AI_SERVICE_API_KEY"),
            queue_region=os.getenv("QUEUE_REGION", "gru1").strip(),
            llm_api_url=_required("LLM_API_URL").rstrip("/"),
            llm_model=_required("LLM_MODEL"),
            llm_api_token=_required("LLM_API_TOKEN"),
            configuration_version=os.getenv(
                "LLM_CONFIGURATION_VERSION", "generative/generar-rutina@1"
            ).strip(),
            timeout_seconds=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "120")),
            max_attempts=retries + 1,
        )
