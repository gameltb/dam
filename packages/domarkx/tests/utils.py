"""Test utilities."""

import typer

from domarkx.cli import cli_app, load_actions
from domarkx.config import settings


def setup_test_app() -> typer.Typer:
    """Set up the test application."""
    load_actions(settings)
    return cli_app
