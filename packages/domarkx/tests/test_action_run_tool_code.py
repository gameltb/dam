import pathlib
from typer.testing import CliRunner
from domarkx.cli import cli_app

runner = CliRunner()

def test_run_tool_code_action(tmp_path: pathlib.Path):
    md_content = """
<message>
<tool_code>
print("Hello from tool code!")
</tool_code>
</message>
"""
    doc_path = tmp_path / "test.md"
    doc_path.write_text(md_content)

    result = runner.invoke(cli_app, ["run-tool-code", str(doc_path), "0"])

    assert result.exit_code == 0
    assert "Hello from tool code!" in result.stdout
