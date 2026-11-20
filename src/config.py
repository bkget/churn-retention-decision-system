"""Load project settings from config/settings.yaml (+ optional .env)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=4)
def load_settings(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return settings as a nested dict. Env var CHURN_DATA_SOURCE overrides data.source."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # python-dotenv is optional at import time
        pass

    with open(path or DEFAULT_SETTINGS, encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)

    override = os.getenv("CHURN_DATA_SOURCE")
    if override:
        settings["data"]["source"] = override
    return settings


def resolve(relative: str | os.PathLike) -> Path:
    """Resolve a path from settings relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
