from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path

import gym_engine.api.app as api_module

ROOT = Path(__file__).resolve().parent.parent


def test_root_entrypoint_reexports_package_app() -> None:
    spec = importlib.util.spec_from_file_location("app_entrypoint", ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.app is api_module.app


def test_vercel_subscriber_entrypoints_are_importable() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    subscribers = pyproject["tool"]["vercel"]["subscribers"]
    assert subscribers, "expected at least one vercel subscriber"

    for subscriber in subscribers:
        entrypoint = subscriber["entrypoint"]
        assert not entrypoint.startswith("src."), (
            f"entrypoint must reference the installed package: {entrypoint}"
        )
        assert importlib.import_module(entrypoint) is not None
