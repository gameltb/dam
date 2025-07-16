import os
from domarkx.config import Settings


def test_settings_default_project_path():
    settings = Settings()
    assert settings.DOMARKX_PROJECT_PATH == "."
    assert settings.project_path == os.path.abspath(".")


def test_settings_custom_project_path(monkeypatch):
    monkeypatch.setenv("DOMARKX_PROJECT_PATH", "/tmp/test_project")
    settings = Settings()
    assert settings.DOMARKX_PROJECT_PATH == "/tmp/test_project"
    assert settings.project_path == os.path.abspath("/tmp/test_project")
