"""Tool registration and execution for running tool calls."""

import io
import logging
from collections.abc import Callable
from typing import Any

from rich.console import Console

from domarkx.tools.apply_diff import apply_diff_tool
from domarkx.tools.attempt_completion import attempt_completion_tool
from domarkx.tools.tool_factory import default_tool_factory

from ...utils.no_border_rich_tracebacks import NoBorderTraceback

logger = logging.getLogger(__name__)

REGISTERED_TOOLS: dict[str, Callable[..., Any]] = {}


# Initialize a Console instance to capture content in a StringIO object
console_output = io.StringIO()
console = Console(file=console_output, soft_wrap=True)


def register_tool(name: str) -> Callable[..., Any]:
    """
    Register a function as an executable tool.

    Args:
        name (str): The name of the tool (corresponds to the XML tag name).

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        REGISTERED_TOOLS[name] = func
        return func

    return decorator


def execute_tool_call(
    tool_call: dict[str, Any], return_traceback: bool = False, handle_exception: bool = True
) -> tuple[Any, str]:
    """
    Execute a parsed tool call.

    Args:
        tool_call (dict): A single tool call dictionary returned by parse_tool_calls.
        return_traceback (bool): Whether to include a traceback in the result upon error.
        handle_exception (bool): Whether to handle exceptions or re-raise them.

    Returns:
        tuple: (tool_name: str, result: str) The result of the tool execution.

    Raises:
        ValueError: If the tool is not found.

    """
    tool_name = tool_call.get("tool_name")
    parameters = tool_call.get("parameters", {})

    logger.info("Attempting to execute tool '%s' with parameters: %s", tool_name, parameters)

    if tool_name not in REGISTERED_TOOLS:
        error_msg = f"Tool '{tool_name}' not found."
        logger.error(error_msg)
        raise ValueError(error_msg)

    tool_func = REGISTERED_TOOLS[tool_name]

    try:
        result = tool_func(**parameters)
        logger.info("Tool '%s' executed successfully.", tool_name)
        return tool_name, str(result)  # Ensure tool name and string result are returned
    except Exception as e:
        error_msg = f"An error occurred while executing tool '{tool_name}': {e}"
        logger.exception(error_msg)  # Use .exception() to log with traceback
        if not handle_exception:
            raise e
        console_output.truncate(0)
        console_output.seek(0)
        console.print(NoBorderTraceback(show_locals=True, extra_lines=1, max_frames=1))
        traceback_str = console_output.getvalue()
        return (
            tool_name,
            f"Error : {error_msg}\n"
            + (
                f"Traceback:\n{'\n'.join([line.rstrip() for line in traceback_str.splitlines()])}"
                if return_traceback
                else ""
            ),
        )


def format_assistant_response(tool_name: str, tool_result: str) -> str:
    """Format the tool execution result into a complete assistant response."""
    return f'<tool_output tool_name="{tool_name}">\n{tool_result}\n</tool_output>'


def register_tools() -> None:
    """Register all available tools."""
    register_tool("apply_diff")(apply_diff_tool)
    register_tool("attempt_completion")(attempt_completion_tool)
    register_tool("execute_command")(default_tool_factory.get_tool("tool_execute_command"))
    register_tool("list_files")(default_tool_factory.get_tool("tool_list_files"))
    register_tool("read_file")(default_tool_factory.get_tool("tool_read_file"))
    register_tool("write_to_file")(default_tool_factory.get_tool("tool_write_to_file"))


register_tools()
