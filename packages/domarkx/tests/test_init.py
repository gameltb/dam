"""Tests for the init command."""

import pathlib

from typer.testing import CliRunner

from domarkx.cli import cli_app

runner = CliRunner()


def test_init_default_path() -> None:
    """Test that the init command creates the default project structure."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli_app, ["init"])
        assert result.exit_code == 0
        assert pathlib.Path(".git").exists()
        assert pathlib.Path("sessions").exists()
        assert pathlib.Path("templates").exists()
        assert pathlib.Path("ProjectManager.md").exists()


def test_init_custom_path() -> None:
    """Test that the init command creates a project structure in a custom path."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli_app, ["init", "--path", "my_project"])
        assert result.exit_code == 0
        assert pathlib.Path("my_project/.git").exists()
        assert pathlib.Path("my_project/sessions").exists()
        assert pathlib.Path("my_project/templates").exists()
        assert pathlib.Path("my_project/ProjectManager.md").exists()
