"""Tests for the CLI application."""

from typer.testing import CliRunner
from utils import setup_test_app

runner = CliRunner()
cli_app = setup_test_app()


def test_app() -> None:
    """Test that the CLI application runs and shows help."""
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "exec-doc" in result.stdout
    assert "exec-doc-code-block" in result.stdout
