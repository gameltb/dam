"""Tests for the run_tool_code action."""

import pathlib

from typer.testing import CliRunner
from utils import setup_test_app

runner = CliRunner()
cli_app = setup_test_app()


def test_run_tool_code_action(tmp_path: pathlib.Path) -> None:
    """Test the run-tool-code action."""
    md_content = """
## user

>hello

## assistant

> I will now print "Hello from tool code!"
```python
print("Hello from tool code!")
```
"""
    doc_path = tmp_path / "test.md"
    doc_path.write_text(md_content)

    result = runner.invoke(cli_app, ["exec-doc-code-block", str(doc_path), "1", "0"], input="y\n")

    assert result.exit_code == 0
    assert "Hello from tool code!" in result.stdout
