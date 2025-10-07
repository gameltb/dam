import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def remote_tool_handler[F: Callable[..., Any]](func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"Tool '{func.__name__}' started.")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Tool '{func.__name__}' completed successfully.")
            return result
        except Exception as e:
            logging.error(f"Tool '{func.__name__}' encountered an error: {e}")
            # Also print traceback to stderr for visibility in Jupyter
            traceback.print_exc()
            raise

    return wrapper  # type: ignore
