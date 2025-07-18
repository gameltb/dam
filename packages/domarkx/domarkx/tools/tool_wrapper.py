import inspect
from typing import Any, Callable

from autogen_core.tools import FunctionTool

from domarkx.tool_executors.jupyter import JupyterToolExecutor


class ToolWrapper(FunctionTool):
    def __init__(self, tool_func: Callable, executor: JupyterToolExecutor, **kwargs):
        description = inspect.getdoc(tool_func) or ""
        super().__init__(tool_func, description=description, **kwargs)
        self.executor = executor

    def call(self, **kwargs: Any) -> Any:
        # Since FunctionTool already validates arguments, we can directly execute.
        return self.executor.execute(self.fn, **kwargs)

    @property
    def source_code(self) -> str:
        """
        Returns the source code of the wrapped tool function.
        """
        if hasattr(self.fn, "__source_code__"):
            return self.fn.__source_code__
        return inspect.getsource(self.fn)
