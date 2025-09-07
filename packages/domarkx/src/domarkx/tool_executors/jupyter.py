import inspect
import json
from typing import Any, Callable, cast

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor

from . import remote_tool_handler


class JupyterToolExecutor:
    def __init__(self, code_executor: CodeExecutor):
        self.code_executor = code_executor

    async def start(self) -> None:
        if hasattr(self.code_executor, "start"):
            await self.code_executor.start()

    async def stop(self) -> None:
        if hasattr(self.code_executor, "stop"):
            await self.code_executor.stop()

    async def restart(self) -> None:
        if hasattr(self.code_executor, "restart"):
            await self.code_executor.restart()

    async def execute(self, tool_func: Callable[..., Any], **kwargs: Any) -> Any:
        if hasattr(tool_func, "__source_code__"):
            tool_source = tool_func.__source_code__  # type: ignore
        else:
            tool_module = inspect.getmodule(tool_func)
            if tool_module is None:
                raise ValueError(f"Could not find module for function {tool_func.__name__}")
            tool_source = inspect.getsource(tool_module)

        tool_name = tool_func.__name__

        remote_tool_handler_module = inspect.getmodule(remote_tool_handler)
        if remote_tool_handler_module is None:
            raise ValueError("Could not find module for remote_tool_handler")
        remote_tool_handler_source = inspect.getsource(remote_tool_handler_module)

        # Serialize arguments to JSON
        serialized_args = json.dumps(kwargs).replace("\\", "\\\\").replace("'", "\\'")

        # Create the code to execute in the Jupyter kernel
        code_to_execute = f"""
{tool_source}

import json

# Deserialize arguments
args = json.loads('''{serialized_args}''')

# Decorate and call the function
{remote_tool_handler_source}
wrapped_tool = remote_tool_handler({tool_name})
result = wrapped_tool(**args)
print(result)
"""

        # Execute the code. autogen_core.CodeExecutor implementations vary; cast to Any
        executor = cast(Any, self.code_executor)
        if hasattr(executor, "execute_code_block"):
            # Some executors provide a sync execute_code_block
            execution_result: Any = executor.execute_code_block(language="python", code=code_to_execute)
        else:
            cancellation_token = CancellationToken()
            code_block = CodeBlock(code=code_to_execute, language="python")
            execution_result: Any = await executor.execute_code_blocks([code_block], cancellation_token)

        if execution_result.exit_code != 0:
            raise RuntimeError(f"Tool execution failed: {execution_result.output}")

        return execution_result.output
