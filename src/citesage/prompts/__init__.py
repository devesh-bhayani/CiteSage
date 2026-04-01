"""Prompt loader — reads YAML files from the versioned prompts directory.

All prompts MUST live in YAML files (never inline strings).  The version
to load is controlled by ``prompts.version`` in config.yaml.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ..config import get_settings


def _prompts_dir() -> Path:
    """Return the absolute path to the active prompt version directory."""
    settings = get_settings()
    base = Path(settings.prompts.path)
    if not base.is_absolute():
        # Resolve relative to project root (walk up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        base = project_root / base
    return base / settings.prompts.version


@lru_cache(maxsize=32)
def load_prompt(name: str) -> dict:
    """Load and cache a prompt YAML file by *name* (without extension).

    Returns the parsed dict so callers can pick out ``system``, ``user``, etc.
    """
    path = _prompts_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
