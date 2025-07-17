import inspect
import json
from typing import Any, Callable, Dict, Optional

from autogen_ext.code_executors.base import CodeExecutor
from autogen_ext.code_executors.docker_jupyter import DockerJupyterCodeExecutor


class JupyterToolExecutor:
    def __init__(self, code_executor: Optional[CodeExecutor] = None):
        self.code_executor = code_executor or DockerJupyterCodeExecutor()

    def start(self):
        if hasattr(self.code_executor, "start"):
            self.code_executor.start()

    def stop(self):
        if hasattr(self.code_executor, "stop"):
            self.code_executor.stop()

    def restart(self):
        if hasattr(self.code_executor, "restart"):
            self.code_executor.restart()

    def execute(self, tool_func: Callable, **kwargs: Any) -> Any:
        tool_source = inspect.getsource(tool_func)
        tool_name = tool_func.__name__

        with open("packages/domarkx/domarkx/tool_executors/remote_tool_handler.py", "r") as f:
            remote_tool_handler_source = f.read()

        # Serialize arguments to JSON
        serialized_args = json.dumps(kwargs)

        # Create the code to execute in the Jupyter kernel
        code_to_execute = f"""
{remote_tool_handler_source}
{tool_source}

# Deserialize arguments
args = json.loads('''{serialized_args}''')

# Decorate and call the function
wrapped_tool = remote_tool_handler({tool_name})
result = wrapped_tool(**args)
print(result)
"""

        # Execute the code
        execution_result = self.code_executor.execute_code_block(language="python", code=code_to_execute)

        if execution_result.exit_code != 0:
            raise RuntimeError(f"Tool execution failed: {execution_result.output}")

        return execution_result.output
