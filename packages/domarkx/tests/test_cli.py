from typer.testing import CliRunner

from domarkx.domarkx.cli import cli_app

runner = CliRunner()


def test_app():
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "exec-doc" in result.stdout
    assert "exec-doc-code-block" in result.stdout
