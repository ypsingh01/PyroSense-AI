"""Shared pytest fixtures for PyroSense AI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _set_test_env() -> None:
    """Force deterministic settings for tests."""

    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./pyrosense_test.db")
    Path("pyrosense_test.db").unlink(missing_ok=True)

