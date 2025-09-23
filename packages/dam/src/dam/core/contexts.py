from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, TypeVar

T = TypeVar("T", covariant=True)


class ContextProvider(Protocol[T]):
    """
    A protocol for providers that can set up and tear down a context.
    """

    def __call__(self, **kwargs: Any) -> AbstractAsyncContextManager[T]: ...
