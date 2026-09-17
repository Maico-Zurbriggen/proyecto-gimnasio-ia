from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Identificador fijo del contrato de salida (RutinaEstructurada), no configuracion de
# deployment -- develop usa el mismo patron (contract_version hardcodeado en service.py),
# y comparten la misma tabla ai_generation_attempts.contract_version, asi que conviene
# el mismo valor para no fragmentar el campo entre dos convenciones distintas.
CONTRACT_VERSION = "routine-generation@1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    app_env: str = "test"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str

    ai_service_api_key: str | None = None

    llm_provider: str = "ollama"  # "ollama" | "openai"

    llm_api_url: str = "http://127.0.0.1:11434"
    # alias LLM_API_TOKEN: asi esta provisionado el secreto real (Cloudflare Tunnel), no
    # LLM_API_KEY.
    llm_api_token: str | None = Field(default=None, validation_alias="LLM_API_TOKEN")
    # alias LLM_MODEL: idem, es el nombre real del secreto para el modelo de Ollama.
    ollama_model: str = Field(default="llama3.1", validation_alias="LLM_MODEL")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    generation_timeout_seconds: int = 120
    generation_max_retries: int = 1
    failed_result_retention_days: int = 30

    poller_interval_seconds: float = 2.0

    # distinto del de develop a proposito: cada implementacion tiene su propio prompt,
    # asi que su version no puede compartir identificador con la de develop sin mentir
    # sobre que produjo el resultado (RF-072).
    configuration_version: str = Field(
        default="generative/generar-rutina@1", validation_alias="LLM_CONFIGURATION_VERSION"
    )

    @property
    def expected_api_key(self) -> str | None:
        """Clave que debe traer el header X-API-Key.

        Una sola clave por deployment (no por app_env adentro del proceso): Vercel ya
        separa test/producción con secretos distintos por environment.
        """
        return self.ai_service_api_key

    @property
    def model_version(self) -> str:
        """Identificador del modelo activo, según LLM_PROVIDER."""
        if self.llm_provider == "openai":
            return self.openai_model
        return self.ollama_model


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
