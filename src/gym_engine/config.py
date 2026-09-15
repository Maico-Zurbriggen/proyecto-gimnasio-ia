from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "test"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str

    ai_service_api_key_test: str | None = None
    ai_service_api_key_production: str | None = None

    llm_provider: str = "ollama"  # "ollama" | "openai"

    llm_api_url: str = "http://127.0.0.1:11434"
    llm_auth_mode: str = "none"
    llm_api_key: str | None = None
    ollama_model: str = "llama3.1"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    generation_timeout_seconds: int = 120
    generation_max_retries: int = 1
    failed_result_retention_days: int = 30

    poller_interval_seconds: float = 2.0

    @property
    def expected_api_key(self) -> str | None:
        """Clave que debe traer el header X-API-Key, según el ambiente."""
        if self.app_env == "production":
            return self.ai_service_api_key_production
        return self.ai_service_api_key_test


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
