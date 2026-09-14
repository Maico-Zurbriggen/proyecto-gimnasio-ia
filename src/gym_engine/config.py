from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "test"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str

    llm_api_url: str = "http://127.0.0.1:11434"
    llm_auth_mode: str = "none"
    llm_api_key: str | None = None
    ollama_model: str = "llama3.1"

    generation_timeout_seconds: int = 120
    generation_max_retries: int = 1
    failed_result_retention_days: int = 30

    poller_interval_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
