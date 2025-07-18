import inspect
import json
from typing import Any, Callable

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor

from . import remote_tool_handler


class JupyterToolExecutor:
    def __init__(self, code_executor: CodeExecutor):
        self.code_executor = code_executor

    def start(self):
        if hasattr(self.code_executor, "start"):
            self.code_executor.start()

    def stop(self):
        if hasattr(self.code_executor, "stop"):
            self.code_executor.stop()

    def restart(self):
        if hasattr(self.code_executor, "restart"):
            self.code_executor.restart()

    async def execute(self, tool_func: Callable, **kwargs: Any) -> Any:
        if hasattr(tool_func, "__source_code__"):
            tool_source = tool_func.__source_code__
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

        # Execute the code
        if hasattr(self.code_executor, "execute_code_block"):
            execution_result = self.code_executor.execute_code_block(language="python", code=code_to_execute)
        else:
            cancellation_token = CancellationToken()
            code_block = CodeBlock(code=code_to_execute, language="python")
            execution_result = await self.code_executor._execute_code_block(code_block, cancellation_token)

        if execution_result.exit_code != 0:
            raise RuntimeError(f"Tool execution failed: {execution_result.output}")

        return execution_result.output
