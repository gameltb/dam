"""Tests for the DAM application's CLI commands."""

import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from typer import Exit
from typer.testing import CliRunner

from dam_app.main import list_worlds, validate_world_and_config

runner = CliRunner()


def test_command_requires_world_and_config(mocker: MockerFixture):
    """Verify that a command fails if no config is found."""
    mock_context = mocker.patch("typer.Context")
    mock_context.invoked_subcommand = "assets"

    with pytest.raises(Exit):
        validate_world_and_config(mock_context, None)


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
    os.environ["DAM_CONFIG_FILE"] = str(dam_toml_path)

    worlds = list_worlds()

    assert "world_one" in worlds
    assert "world_two" in worlds
