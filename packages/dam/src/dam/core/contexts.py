from typing import Any, AsyncGenerator, Protocol, TypeVar

T = TypeVar("T")


class ContextProvider(Protocol[T]):
    """
    A protocol for providers that can set up and tear down a context.
    """

    async def __call__(self, **kwargs: Any) -> AsyncGenerator[T, None]:
        ...
