import pytest

from domarkx.autogen_session import AutoGenSession


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

    session = AutoGenSession(doc_path)

    assert session.doc is not None


def test_create_session_with_override_parameters(tmp_path):
    doc_content = """
[@my-variable](domarkx://set?value=default)

```python setup-script
from unittest.mock import MagicMock
client = MagicMock()
tools = []
```
"""
    doc_path = tmp_path / "test.md"
    doc_path.write_text(doc_content)

    override_parameters = {"my-variable": {"value": "overridden"}}
    session = AutoGenSession(doc_path, override_parameters=override_parameters)
    rendered_markdown = session.doc.to_markdown()
    assert "overridden" in rendered_markdown
    assert "[@my-variable]" not in rendered_markdown


@pytest.mark.asyncio
async def test_session_setup_async(tmp_path, setup_script):
    doc_path = tmp_path / "test.md"
    doc_path.write_text(setup_script)

    session = AutoGenSession(doc_path)
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

    session = AutoGenSession(doc_path)

    assert session.get_code_block("foo").rstrip() == 'print("foo")'
    assert session.get_code_block("bar").rstrip() == 'print("bar")'
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

    session = AutoGenSession(doc_path)
    await session.setup()

    assert session.agent is not None
    assert len(session.tool_executors) == 0
    assert session.get_code_block("my-tool") == 'print(get_code_block("my-tool"))\n'
