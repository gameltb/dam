from typing import Any, AsyncGenerator, List

from dam.core.system_events import BaseSystemEvent, SystemResultEvent


class CommandStream:
    """
    A wrapper around the async generator returned by dispatch_command.

    This class provides convenience methods to consume the stream and extract
    results, while also allowing the stream to be iterated over directly.
    """

    def __init__(self, generator: AsyncGenerator[BaseSystemEvent, None]):
        self._generator = generator
        self._results: List[Any] = []
        self._consumed = False

    def __aiter__(self) -> "CommandStream":
        return self

    async def __anext__(self) -> BaseSystemEvent:
        """
        Makes CommandStream itself an async iterator.
        """
        if self._consumed:
            raise StopAsyncIteration
        try:
            event = await self._generator.__anext__()
            if isinstance(event, SystemResultEvent):
                self._results.append(event.result)  # type: ignore
            return event  # type: ignore
        except StopAsyncIteration:
            self._consumed = True
            raise

    async def _consume_all(self) -> None:
        """Helper to consume the stream if it hasn't been already."""
        if not self._consumed:
            async for _ in self:
                # Just consume to populate self._results
                pass

    async def get_all_results(self) -> List[Any]:
        """
        Consumes the stream and returns a list of all results from SystemResultEvents.
        """
        await self._consume_all()
        return self._results

    async def get_one_value(self) -> Any:
        """
        Consumes the stream and returns the single result value.
        Raises a ValueError if there is not exactly one result.
        """
        await self._consume_all()
        if len(self._results) != 1:
            raise ValueError(f"Expected one result, but found {len(self._results)}.")
        return self._results[0]

    async def get_first_non_none_value(self) -> Any:
        """
        Consumes the stream and returns the first result that is not None.
        Raises a ValueError if no non-None result is found.
        """
        await self._consume_all()
        for res in self._results:
            if res is not None:
                return res
        raise ValueError("No non-None results found.")
