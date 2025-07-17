import functools
import inspect
import io
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback

# Global console instance for rich output, always available
# We use a StringIO to capture rich-formatted logs for later retrieval
captured_rich_logs_buffer = io.StringIO()
# The console now writes to both the buffer and stderr
console = Console(file=captured_rich_logs_buffer, stderr=True)

# Global registry for tool_handler-decorated functions
TOOL_REGISTRY = []


class ToolError(Exception):
    """Custom exception for tool-related errors."""

    def __init__(self, message, original_exception=None, captured_logs_str=""):
        super().__init__(message)
        self.original_exception = original_exception
        self.captured_logs = captured_logs_str  # This will now contain rich-formatted logs


def tool_handler(log_level=logging.INFO, capture_logs=False):
    """
    A decorator for tool functions to handle logging and exception wrapping.

    Args:
        log_level (int): The logging level for messages from the decorated function.
                         Defaults to logging.INFO. Can be changed at runtime via
                         the 'log_level' argument in the decorated function's call.
        capture_logs (bool): Whether to capture logs and include them in ToolError. Defaults to False.
                             When True, logs are captured in Rich's formatted style.
    """

    def decorator(func):
        # Register the function in the global registry
        TOOL_REGISTRY.append(func)

        # Check for parameter name conflicts
        tool_handler_params = {"log_level", "capture_logs", "show_locals", "show_stacktrace"}
        func_signature = inspect.signature(func)
        for param_name in func_signature.parameters:
            if param_name in tool_handler_params:
                logging.warning(
                    f"Warning: Parameter name '{param_name}' in function '{func.__name__}' "
                    f"conflicts with a reserved parameter name for tool_handler decorator. "
                    f"The decorator's parameter will take precedence."
                )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Allow overriding log_level, capture_logs, show_locals, show_stacktrace at runtime
            runtime_log_level = kwargs.pop("log_level", log_level)
            runtime_capture_logs = kwargs.pop("capture_logs", capture_logs)
            show_locals = kwargs.pop("show_locals", False)
            show_stacktrace = kwargs.pop("show_stacktrace", False)

            original_level = logging.getLogger().level
            rich_handler = None

            # Clear the buffer at the start of each tool call to ensure fresh logs
            captured_rich_logs_buffer.seek(0)
            captured_rich_logs_buffer.truncate(0)

            # Configure logger for this execution
            logging.getLogger().setLevel(runtime_log_level)

            # Use RichHandler for console output and internal buffer capture
            rich_handler = RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
                log_time_format="%H:%M:%S",
            )
            logging.getLogger().addHandler(rich_handler)
            logging.getLogger().propagate = (
                False  # Prevent logs from propagating to default handlers (e.g., basicConfig)
            )

            logging.info(f"Tool '{func.__name__}' started.")

            captured_logs_str = ""  # Initialize for captured logs
            try:
                result = func(*args, **kwargs)
                logging.info(f"Tool '{func.__name__}' completed successfully.")

                # If capture_logs is True, get the rich-formatted logs from the buffer
                if runtime_capture_logs:
                    captured_logs_str = captured_rich_logs_buffer.getvalue().strip()

                return result  # Directly return result on success, logs are not returned as a separate value on success
            except Exception as e:
                # Always capture logs on exception if capture_logs is enabled for the decorator
                if runtime_capture_logs:
                    captured_logs_str = captured_rich_logs_buffer.getvalue().strip()

                # Format the exception details into a string using Rich's Traceback
                error_output = io.StringIO()
                # Use a temporary console to print traceback to a string, preventing interference with main console
                temp_traceback_console = Console(file=error_output, record=True)
                temp_traceback_console.print(
                    Traceback(
                        show_locals=show_locals,
                        suppress=[__file__, functools.__file__],  # Suppress decorator internal frames
                        extra_lines=3,  # Adjust as needed for context
                        width=temp_traceback_console.width,  # Use traceback console's width
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

                logging.error(f"Tool '{func.__name__}' encountered an error. See ToolError details for full report.")
                raise ToolError(full_error_message, original_exception=e, captured_logs_str=captured_logs_str)
            finally:
                # Clean up: remove handler and restore original log level
                if rich_handler:
                    logging.getLogger().removeHandler(rich_handler)
                    logging.getLogger().propagate = True  # Restore propagation
                logging.getLogger().setLevel(original_level)

        return wrapper

    return decorator
