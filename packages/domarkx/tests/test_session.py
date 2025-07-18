import pytest
from pathlib import Path
from domarkx.session import Session


@pytest.fixture
def setup_script():
    return """
```python setup-script
from unittest.mock import MagicMock
client = MagicMock()
def my_tool():
    pass
tools = [my_tool]
```
"""


def test_session_setup(tmp_path, setup_script):
    doc_path = tmp_path / "test.md"
    doc_path.write_text(setup_script)

    session = Session(doc_path)

    assert session.doc is not None


@pytest.mark.asyncio
async def test_session_setup_async(tmp_path, setup_script):
    doc_path = tmp_path / "test.md"
    doc_path.write_text(setup_script)

    session = Session(doc_path)
    await session.setup()

    assert session.agent is not None
    assert len(session.tool_executors) == 0


def test_get_code_block(tmp_path):
    doc_content = """
```python foo
print("foo")
```

```python bar
print("bar")
```
"""
    doc_path = tmp_path / "test.md"
    doc_path.write_text(doc_content)

    session = Session(doc_path)

    assert session.get_code_block("foo") == 'print("foo")'
    assert session.get_code_block("bar") == 'print("bar")'
    assert session.get_code_block("baz") is None


@pytest.mark.asyncio
async def test_remote_tool_execution(tmp_path):
    doc_content = """
```python setup-script
from unittest.mock import MagicMock
client = MagicMock()

def remote_tool():
    return "remote tool executed"

tools = [remote_tool]
```

```python my-tool
print(get_code_block("my-tool"))
```
"""
    doc_path = tmp_path / "test.md"
    doc_path.write_text(doc_content)

    session = Session(doc_path)
    await session.setup()

    assert session.agent is not None
    assert len(session.tool_executors) == 0
    assert session.get_code_block("my-tool") == 'print(get_code_block("my-tool"))\n'
