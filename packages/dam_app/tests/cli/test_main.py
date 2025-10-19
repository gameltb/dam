"""Tests for the DAM application's CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from dam_app.main import app

runner = CliRunner()


def test_list_worlds_command(tmp_path: Path):
    """Verify that the list-worlds command correctly lists worlds from a config file."""
    # Create a dummy dam.toml
    dam_toml_path = tmp_path / "dam.toml"
    dam_toml_path.write_text(
        """
[worlds.world_one.plugin_settings.core]
database_url="sqlite:///one.db"
alembic_path="./migrations_one"

[worlds.world_two.plugin_settings.core]
database_url="sqlite:///two.db"
alembic_path="./migrations_two"
"""
    )

    result = runner.invoke(app, ["--config", str(dam_toml_path), "list-worlds"])

    assert result.exit_code == 0
    assert "world_one" in result.stdout
    assert "world_two" in result.stdout


def test_command_requires_world_and_config():
    """Verify that a command fails if no config is found."""
    # Running in an isolated filesystem ensures no dam.toml is found
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["assets", "list", "--world", "non_existent_world"])

        assert result.exit_code == 1
        assert "Error: Failed to instantiate world" in result.stdout
