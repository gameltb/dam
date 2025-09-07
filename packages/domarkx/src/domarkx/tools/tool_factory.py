import functools
import importlib
import inspect
import io
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from autogen_core.tools import FunctionTool
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback


class ToolError(Exception):
    """Custom exception for tool-related errors."""

    def __init__(
        self, message: str, original_exception: Optional[Exception] = None, captured_logs_str: str = ""
    ) -> None:
        super().__init__(message)
        self.original_exception = original_exception
        self.captured_logs = captured_logs_str


def tool_handler(log_level: int = logging.INFO, capture_logs: bool = False) -> Callable[..., Any]:
    """
    A decorator for tool functions to handle logging and exception wrapping.
    This is a private function used by the ToolFactory.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            captured_rich_logs_buffer = io.StringIO()
            console = Console(file=captured_rich_logs_buffer, stderr=True)
            runtime_log_level = kwargs.pop("log_level", log_level)
            runtime_capture_logs = kwargs.pop("capture_logs", capture_logs)
            show_locals = kwargs.pop("show_locals", False)

            original_level = logging.getLogger().level
            rich_handler = RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
                log_time_format="%H:%M:%S",
            )
            logging.getLogger().addHandler(rich_handler)
            logging.getLogger().setLevel(runtime_log_level)
            logging.getLogger().propagate = False

            logging.info(f"Tool '{func.__name__}' started.")
            try:
                result = func(*args, **kwargs)
                logging.info(f"Tool '{func.__name__}' completed successfully.")
                captured_logs_str = ""
                if runtime_capture_logs:
                    captured_logs_str = captured_rich_logs_buffer.getvalue().strip()
                return result
            except Exception as e:
                captured_logs_str = captured_rich_logs_buffer.getvalue().strip()
                error_output = io.StringIO()
                temp_traceback_console = Console(file=error_output, record=True)
                temp_traceback_console.print(
                    Traceback(
                        show_locals=show_locals,
                        suppress=[__file__, functools.__file__],
                        extra_lines=3,
                        width=temp_traceback_console.width,
                    )
                )
                formatted_exception = error_output.getvalue().strip()
                full_error_message = f"""
An error occurred in tool '{func.__name__}':

--- Error Details ---
{formatted_exception}

--- Captured Logs ---
{captured_logs_str}
""".strip()
                logging.error(f"Tool '{func.__name__}' encountered an error.")
                raise ToolError(full_error_message, original_exception=e, captured_logs_str=captured_logs_str)
            finally:
                logging.getLogger().removeHandler(rich_handler)
                logging.getLogger().propagate = True
                logging.getLogger().setLevel(original_level)

        return wrapper

    return decorator


class ToolFactory:
    def __init__(self) -> None:
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        self._tool_registry_unwrapped: Dict[str, Callable[..., Any]] = {}
        self._discover_and_register_tools()

    def _discover_and_register_tools(self) -> None:
        """
        Dynamically discovers and registers tools from the 'portable_tools' directory.
        """
        portable_tools_path = os.path.join(os.path.dirname(__file__), "portable_tools")
        for root, _, files in os.walk(portable_tools_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__init__"):
                    module_path = os.path.join(root, file)
                    module_name = os.path.splitext(os.path.relpath(module_path, portable_tools_path))[0].replace(
                        os.sep, "."
                    )
                    module = importlib.import_module(f"domarkx.tools.portable_tools.{module_name}")

                    with open(module_path, "r") as f:
                        module_source = f.read()

                    for attribute_name, attribute in inspect.getmembers(module, inspect.isfunction):
                        if attribute_name.startswith("tool_"):
                            self._tool_registry_unwrapped[attribute_name] = attribute
                            wrapped_tool = tool_handler()(attribute)
                            wrapped_tool.__source_code__ = module_source
                            self._tool_registry[attribute_name] = wrapped_tool

    def get_tool(self, name: str) -> Callable[..., Any]:
        """
        Retrieves a tool from the registry by its name.
        """
        return self._tool_registry[name]

    def get_unwrapped_tool(self, name: str) -> Callable[..., Any]:
        """
        Retrieves an unwrapped tool from the registry by its name.
        """
        return self._tool_registry_unwrapped[name]

    def list_tools(self) -> List[Dict[str, Optional[str]]]:
        """
        Lists all available tool functions registered in the tool registry.
        """
        tool_infos: List[Dict[str, Optional[str]]] = []
        for name, func in self._tool_registry.items():
            tool_infos.append(
                {
                    "name": name,
                    "doc": inspect.getdoc(func),
                    "signature": str(inspect.signature(func)),
                }
            )
        return tool_infos

    def wrap_function(self, func: Callable[..., Any], executor: Optional[Any] = None) -> FunctionTool:
        """
        Wraps a given function to be used as a tool, with an optional executor.
        """
        wrapped_func = tool_handler()(func)

        class DynamicToolWrapper(FunctionTool):
            def __init__(self, tool_func: Callable[..., Any], tool_executor: Optional[Any], **kwargs: Any) -> None:
                description = inspect.getdoc(tool_func) or ""
                super().__init__(tool_func, description=description, **kwargs)
                self.executor = tool_executor
                self.fn: Callable[..., Any] = tool_func

            def call(self, **kwargs: Any) -> Any:
                if self.executor:
                    return self.executor.execute(self.fn, **kwargs)
                return self.fn(**kwargs)

            @property
            def source_code(self) -> str:
                if hasattr(self.fn, "__source_code__"):
                    return str(self.fn.__source_code__)
                try:
                    return inspect.getsource(self.fn)
                except TypeError:
                    return "Source code not available."

        return DynamicToolWrapper(tool_func=wrapped_func, tool_executor=executor)

    def create_tool_from_string(self, code: str, executor: Optional[Any] = None) -> FunctionTool:
        """
        Creates a tool from a string of Python code.
        """
        from domarkx.utils.code_execution import execute_code_block

        local_namespace: dict[str, Any] = {}
        execute_code_block(code, global_vars=globals(), local_vars=local_namespace)

        func_name = [name for name, obj in local_namespace.items() if inspect.isfunction(obj)][0]
        func = local_namespace[func_name]

        # Attach the source code to the function object
        func.__source_code__ = code

        return self.wrap_function(func, executor)


default_tool_factory = ToolFactory()
