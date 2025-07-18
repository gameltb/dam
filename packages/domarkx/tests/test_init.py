import os

from typer.testing import CliRunner

from domarkx.cli import cli_app

runner = CliRunner()


def test_init_default_path():
    with runner.isolated_filesystem():
        result = runner.invoke(cli_app, ["init"])
        assert result.exit_code == 0
        assert os.path.exists(".git")
        assert os.path.exists("sessions")
        assert os.path.exists("templates")
        assert os.path.exists("ProjectManager.md")


def test_init_custom_path():
    with runner.isolated_filesystem():
        result = runner.invoke(cli_app, ["init", "--path", "my_project"])
        assert result.exit_code == 0
        assert os.path.exists("my_project/.git")
        assert os.path.exists("my_project/sessions")
        assert os.path.exists("my_project/templates")
        assert os.path.exists("my_project/ProjectManager.md")
