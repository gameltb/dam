import asyncio
import pathlib

import pytest
from autogen_core.code_executor import CodeExecutor
from autogen_ext.code_executors.docker_jupyter import DockerJupyterCodeExecutor
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor

from domarkx.tool_executors.jupyter import JupyterToolExecutor
from domarkx.tools.tool_factory import default_tool_factory


# A sample tool function to be executed remotely
def sample_tool(a: int, b: int) -> int:
    return a + b


# Fixture to initialize and finalize code executors
from typing import Any, AsyncGenerator


@pytest.fixture(params=[JupyterCodeExecutor, DockerJupyterCodeExecutor])
async def code_executor(request: Any) -> AsyncGenerator[CodeExecutor, None]:
    executor_class = request.param
    try:
        executor = executor_class()
        if asyncio.iscoroutinefunction(executor.start):
            await executor.start()
        else:
            executor.start()
        yield executor
        if asyncio.iscoroutinefunction(executor.stop):
            await executor.stop()
        else:
            executor.stop()
    except Exception as e:
        pytest.skip(f"Could not start {executor_class.__name__}: {e}")


async def test_jupyter_tool_executor_with_different_code_executors(code_executor: CodeExecutor) -> None:
    tool_executor = JupyterToolExecutor(code_executor=code_executor)

    # Execute the tool
    result = await tool_executor.execute(sample_tool, a=5, b=10)

    # The result is returned as a string, so we need to parse it
    # and check if it matches the expected output.
    # The output format is "exit_code\noutput"
    assert "15" in result


async def test_execute_command_tool(code_executor: CodeExecutor) -> None:
    tool_executor = JupyterToolExecutor(code_executor=code_executor)
    result = await tool_executor.execute(
        default_tool_factory.get_unwrapped_tool("tool_execute_command"), command="echo 'Hello from command'"
    )
    assert "Hello from command" in result


async def test_read_and_write_file_tools(code_executor: CodeExecutor, tmp_path: pathlib.Path) -> None:
    tool_executor = JupyterToolExecutor(code_executor=code_executor)

    file_path = tmp_path / "test_file.txt"
    content = """This is a test file.
It contains multiple lines.
And triple quotes: \"\"\"
"""

    # Test write_to_file
    await tool_executor.execute(
        default_tool_factory.get_unwrapped_tool("tool_write_to_file"), path=str(file_path), content=content
    )

    # Test read_file
    result = await tool_executor.execute(default_tool_factory.get_unwrapped_tool("tool_read_file"), path=str(file_path))

    assert content in result
