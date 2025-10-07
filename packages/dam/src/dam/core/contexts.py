"""Core components for context management."""

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class ContextProvider(Protocol[T_co]):
    """A protocol for providers that can set up and tear down a context."""

    def __call__(self, **kwargs: Any) -> AbstractAsyncContextManager[T_co]:
        """Create and return an async context manager for the context."""
        ...
