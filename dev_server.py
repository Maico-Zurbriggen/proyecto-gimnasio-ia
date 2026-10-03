from __future__ import annotations

import asyncio
import os
from pathlib import Path


def selector_loop_factory(*, use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    del use_subprocess
    return asyncio.SelectorEventLoop()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import uvicorn

    uvicorn.run(
        "gym_engine.api.app:app",
        app_dir="src",
        loop="dev_server:selector_loop_factory" if os.name == "nt" else "auto",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
