from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from gym_engine.api.auth import RequireApiKey
from gym_engine.api.routes import router
from gym_engine.config import Settings, get_settings
from gym_engine.persistence.db import get_connection


def create_app() -> FastAPI:
    app = FastAPI(title="Proyecto Gimnasio — servicio de IA", version="0.1.0")
    app.include_router(router)

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"], dependencies=[RequireApiKey])
    def ready(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
        try:
            conn = get_connection(settings)
            try:
                conn.execute("SELECT 1")
            finally:
                conn.close()
        except Exception as error:
            raise HTTPException(status_code=503, detail="dependency_unavailable") from error
        return {"status": "ready", "database": "up"}

    return app


app = create_app()
