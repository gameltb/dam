import pytest
import asyncio
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_ext.code_executors.docker_jupyter import DockerJupyterCodeExecutor

# A sample tool function to be executed remotely
def sample_tool(a: int, b: int) -> int:
    return a + b

# Fixture to initialize and finalize code executors
@pytest.fixture(params=[JupyterCodeExecutor, DockerJupyterCodeExecutor])
async def code_executor(request):
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

async def test_jupyter_tool_executor_with_different_code_executors(code_executor):
    tool_executor = JupyterToolExecutor(code_executor=code_executor)

    # Execute the tool
    result = await tool_executor.execute(sample_tool, a=5, b=10)

    # The result is returned as a string, so we need to parse it
    # and check if it matches the expected output.
    # The output format is "exit_code\noutput"
    assert "15" in result
