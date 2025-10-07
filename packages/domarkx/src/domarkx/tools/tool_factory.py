"""A factory for creating and managing tools."""

import functools
import importlib
import inspect
import io
import logging
import pathlib
from collections.abc import Callable
from typing import Any

from autogen_core.tools import FunctionTool
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback

from domarkx.utils.code_execution import execute_code_block

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Custom exception for tool-related errors."""

    def __init__(self, message: str, original_exception: Exception | None = None, captured_logs_str: str = "") -> None:
        """Initialize the ToolError."""
        super().__init__(message)
        self.original_exception = original_exception
        self.captured_logs = captured_logs_str


def tool_handler(log_level: int = logging.INFO, capture_logs: bool = False) -> Callable[..., Any]:
    """
    Decorate a tool function to handle logging and exception wrapping.

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

            logger.info("Tool '%s' started.", func.__name__)
            try:
                result = func(*args, **kwargs)
                logger.info("Tool '%s' completed successfully.", func.__name__)
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
                logger.error("Tool '%s' encountered an error.", func.__name__)
                raise ToolError(full_error_message, original_exception=e, captured_logs_str=captured_logs_str) from e
            finally:
                logging.getLogger().removeHandler(rich_handler)
                logging.getLogger().propagate = True
                logging.getLogger().setLevel(original_level)

        return wrapper

    return decorator


class ToolFactory:
    """A factory for creating and managing tools."""

    def __init__(self) -> None:
        """Initialize the ToolFactory."""
        self._tool_registry: dict[str, Callable[..., Any]] = {}
        self._tool_registry_unwrapped: dict[str, Callable[..., Any]] = {}
        self._discover_and_register_tools()

    def _discover_and_register_tools(self) -> None:
        """Dynamically discover and register tools from the 'portable_tools' directory."""
        portable_tools_path = pathlib.Path(__file__).parent / "portable_tools"
        for file in portable_tools_path.rglob("*.py"):
            if not file.name.startswith("__init__"):
                module_path = file
                module_name = ".".join(file.relative_to(portable_tools_path).parts).removesuffix(".py")
                module = importlib.import_module(f"domarkx.tools.portable_tools.{module_name}")

                module_source = module_path.read_text()

                for attribute_name, attribute in inspect.getmembers(module, inspect.isfunction):
                    if attribute_name.startswith("tool_"):
                        self._tool_registry_unwrapped[attribute_name] = attribute
                        wrapped_tool = tool_handler()(attribute)
                        wrapped_tool.__source_code__ = module_source
                        self._tool_registry[attribute_name] = wrapped_tool

    def get_tool(self, name: str) -> Callable[..., Any]:
        """Retrieve a tool from the registry by its name."""
        return self._tool_registry[name]

    def get_unwrapped_tool(self, name: str) -> Callable[..., Any]:
        """Retrieve an unwrapped tool from the registry by its name."""
        return self._tool_registry_unwrapped[name]

    def list_tools(self) -> list[dict[str, str | None]]:
        """List all available tool functions registered in the tool registry."""
        tool_infos: list[dict[str, str | None]] = []
        for name, func in self._tool_registry.items():
            tool_infos.append(
                {
                    "name": name,
                    "doc": inspect.getdoc(func),
                    "signature": str(inspect.signature(func)),
                }
            )
        return tool_infos

    def wrap_function(self, func: Callable[..., Any], executor: Any | None = None) -> FunctionTool:
        """Wrap a given function to be used as a tool, with an optional executor."""
        wrapped_func = tool_handler()(func)

        class DynamicToolWrapper(FunctionTool):
            def __init__(self, tool_func: Callable[..., Any], tool_executor: Any | None, **kwargs: Any) -> None:
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
                    return str(self.fn.__source_code__)  # pyright: ignore[reportFunctionMemberAccess]
                try:
                    return inspect.getsource(self.fn)
                except TypeError:
                    return "Source code not available."

        return DynamicToolWrapper(tool_func=wrapped_func, tool_executor=executor)

    def create_tool_from_string(self, code: str, executor: Any | None = None) -> FunctionTool:
        """Create a tool from a string of Python code."""
        local_namespace: dict[str, Any] = {}
        execute_code_block(code, global_vars=globals(), local_vars=local_namespace)

        func_name = next(name for name, obj in local_namespace.items() if inspect.isfunction(obj))
        func = local_namespace[func_name]

        # Attach the source code to the function object
        func.__source_code__ = code

        return self.wrap_function(func, executor)


default_tool_factory = ToolFactory()
