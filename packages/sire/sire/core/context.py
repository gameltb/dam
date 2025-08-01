import contextlib
import contextvars
from typing import Any, Dict, Optional

# A context variable to hold the arguments for the current inference call.
# This allows decoupled components to access runtime information without
# needing it to be passed manually through multiple layers.
sire_inference_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "sire_inference_context", default=None
)


def get_sire_inference_context() -> Optional[Dict[str, Any]]:
    """
    Retrieves the inference context for the current task.

    Returns:
        A dictionary containing the 'args' and 'kwargs' of the current
        inference call, or None if not in an inference context.
    """
    return sire_inference_context.get()


@contextlib.contextmanager
def sire_inference_context_manager(*args, **kwargs):
    """
    A context manager to set the inference context for the duration of a `with` block.
    """
    token = sire_inference_context.set({"args": args, "kwargs": kwargs})
    try:
        yield
    finally:
        sire_inference_context.reset(token)
