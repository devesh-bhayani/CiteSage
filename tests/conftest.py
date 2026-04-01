"""Root-level pytest configuration.

Sets the working directory to the project root before any tests run so that
get_settings() can find config.yaml by walking up from cwd.
Also clears the lru_cache on get_settings between test sessions to avoid
cross-test contamination when CITESAGE_CONFIG is overridden.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Project root is two levels up from this file (tests/conftest.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True, scope="session")
def set_project_root_as_cwd(tmp_path_factory):
    """Change cwd to project root for the entire test session."""
    original = Path.cwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(original)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the get_settings LRU cache before each test.

    Prevents a test that sets CITESAGE_CONFIG from poisoning later tests.
    """
    from citesage.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
