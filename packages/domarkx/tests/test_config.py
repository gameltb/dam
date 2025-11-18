"""Tests for the configuration settings."""

import pathlib
from typing import Any

from domarkx.config import Settings


def test_settings_default_project_path() -> None:
    """Test that the default project path is the current directory."""
    settings = Settings()
    assert settings.DOMARKX_PROJECT_PATH == "."
    assert settings.project_path == pathlib.Path().resolve()


def test_settings_custom_project_path(monkeypatch: Any) -> None:
    """Test that a custom project path can be set via environment variables."""
    monkeypatch.setenv("DOMARKX_PROJECT_PATH", "/tmp/test_project")
    settings = Settings()
    assert settings.DOMARKX_PROJECT_PATH == "/tmp/test_project"
    assert settings.project_path == pathlib.Path("/tmp/test_project").resolve()
