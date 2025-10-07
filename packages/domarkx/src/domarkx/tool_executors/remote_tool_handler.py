"""A remote tool handler that wraps a tool function to provide logging and error handling."""

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def remote_tool_handler[F: Callable[..., Any]](func: F) -> F:
    """
    Decorate a tool function to handle logging and exception wrapping.

    Args:
        func: The tool function to wrap.

    Returns:
        The wrapped tool function.

    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("Tool '%s' started.", func.__name__)
        try:
            result = func(*args, **kwargs)
            logger.info("Tool '%s' completed successfully.", func.__name__)
            return result
        except Exception as e:
            logger.error("Tool '%s' encountered an error: %s", func.__name__, e)
            # Also print traceback to stderr for visibility in Jupyter
            traceback.print_exc()
            raise

    return wrapper  # type: ignore
