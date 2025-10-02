# pyright: basic
import asyncio
from typing import Any, AsyncGenerator, Generic, List, Optional, Self, TypeVar, overload

from dam.core.requests import InformationRequest
from dam.enums import ExecutionStrategy
from dam.system_events.base import BaseSystemEvent, SystemResultEvent

ResultType = TypeVar("ResultType")
EventType = TypeVar("EventType", bound=BaseSystemEvent)
ItemType = TypeVar("ItemType")
SendType = TypeVar("SendType")


class SystemExecutor(Generic[ResultType, EventType]):
    """
    Executes a collection of system generators according to a specified strategy
    and provides methods to consume the resulting event stream.
    """

    def __init__(
        self,
        generators: List[AsyncGenerator[EventType, Any]],
        strategy: ExecutionStrategy,
    ):
        self._generators = generators
        self._strategy = strategy
        self._results: Optional[List[ResultType]] = None
        self._iterator: Optional[AsyncGenerator[EventType | InformationRequest[Any], Any]] = None

    def __aiter__(self) -> AsyncGenerator[EventType | InformationRequest[Any], Any]:
        # To prevent re-running the generators, we create the iterator once and reuse it.
        if self._iterator is None:
            if self._strategy == ExecutionStrategy.SERIAL:
                self._iterator = self._run_serial()
            elif self._strategy == ExecutionStrategy.PARALLEL:
                self._iterator = self._run_parallel()
            else:
                raise ValueError(f"Unknown execution strategy: {self._strategy}")
        return self._iterator

    async def _run_serial(self) -> AsyncGenerator[EventType | InformationRequest[Any], Any]:
        """Executes generators one by one, handling InformationRequests."""
        for gen in self._generators:
            value_to_send: Any = None
            while True:
                try:
                    # The first asend() must be with None.
                    event = await gen.asend(value_to_send)
                    # Yield the event and receive the next value to send from the consumer.
                    value_to_send = yield event
                except StopAsyncIteration:
                    break

    async def _run_parallel(self) -> AsyncGenerator[EventType | InformationRequest[Any], Any]:
        """Executes generators concurrently, yielding events as they become available."""
        # A queue for events and requests from drainers to the main loop.
        q: asyncio.Queue[EventType | Exception | InformationRequest[Any]] = asyncio.Queue()
        # A single queue for responses from the main loop back to the waiting drainer.
        response_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        # A lock to ensure only one drainer can issue a request and wait for a response at a time.
        response_lock = asyncio.Lock()

        async def drain(gen: AsyncGenerator[EventType, Any]) -> None:
            """Drains a generator's items into a queue, handling requests."""
            value_to_send: Any = None
            while True:
                try:
                    event = await gen.asend(value_to_send)
                    if isinstance(event, InformationRequest):
                        # This drainer needs to make a request.
                        async with response_lock:
                            await q.put(event)
                            # Wait for the response from the main loop.
                            value_to_send = await response_q.get()
                            response_q.task_done()
                    else:
                        await q.put(event)
                        value_to_send = None  # Reset for next iteration
                except StopAsyncIteration:
                    break
                except Exception as e:
                    # If a generator fails, put the exception in the queue to be re-raised.
                    await q.put(e)
                    break

        # Start tasks to drain all generators into the queue.
        drain_tasks = [asyncio.create_task(drain(gen)) for gen in self._generators]

        # Keep track of finished tasks to know when to stop.
        finished_tasks = 0
        while finished_tasks < len(drain_tasks):
            # Wait for an item to appear in the queue.
            item = await q.get()
            q.task_done()

            if isinstance(item, Exception):
                # If an exception was raised, cancel other drain tasks and re-raise.
                for task in drain_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*drain_tasks, return_exceptions=True)
                raise item

            if isinstance(item, InformationRequest):
                # Yield the request up to the main consumer.
                response = yield item
                # Put the response on the queue for the waiting drainer.
                await response_q.put(response)
            else:
                # It's a regular event. The value sent back by the consumer will be
                # received by `yield item` here, but we don't need to do anything with it.
                yield item

            # Check for finished tasks.
            finished_tasks = sum(1 for task in drain_tasks if task.done())

        # Ensure all background tasks are awaited to propagate any final exceptions.
        await asyncio.gather(*drain_tasks)

    async def _populate_results_if_needed(self) -> None:
        """Consumes the internal iterator and stores the results."""
        if self._results is None:
            self._results = []
            async for event in self:
                if isinstance(event, InformationRequest):
                    raise TypeError(
                        "SystemExecutor yielded an InformationRequest, but it was consumed by a method "
                        "that cannot handle it (e.g., get_all_results). You must iterate over the "
                        "executor manually to handle the request."
                    )
                if isinstance(event, SystemResultEvent):
                    self._results.append(event.result)

    async def get_all_results(self) -> List[ResultType]:
        """
        Consumes the stream and returns a list of all results from SystemResultEvents.
        """
        await self._populate_results_if_needed()
        return self._results if self._results is not None else []

    async def get_one_value(self) -> ResultType:
        """
        Consumes the stream and returns the single result value.
        Raises a ValueError if there is not exactly one result.
        """
        results = await self.get_all_results()
        if len(results) != 1:
            raise ValueError(f"Expected one result, but found {len(results)}.")
        return results[0]

    async def get_first_non_none_value(self) -> Optional[ResultType]:
        """
        Consumes the stream until the first non-None result is found and returns it.
        Returns None if no non-None result is found.
        """
        if self._results is not None:
            return next((res for res in self._results if res is not None), None)

        # Optimization: iterate without storing all results if the value is found early.
        async for event in self:
            if isinstance(event, SystemResultEvent) and event.result is not None:
                return event.result  # type: ignore

        # If we get here, the stream is fully consumed, and no non-None value was found.
        # We can now cache the (empty) results list.
        await self._populate_results_if_needed()
        return None

    @overload
    async def get_all_results_flat(self: "SystemExecutor[Optional[List[ItemType]], EventType]") -> List[ItemType]: ...

    @overload
    async def get_all_results_flat(self: "SystemExecutor[List[ItemType], EventType]") -> List[ItemType]: ...

    @overload
    async def get_all_results_flat(self: Self) -> List[ResultType]: ...

    async def get_all_results_flat(self: Self) -> List[Any]:
        """
        Consumes the stream, gets all results, and flattens any that are lists.
        If a result is None, it is ignored.
        """
        results = await self.get_all_results()
        flat_list: List[Any] = []
        for item in results:
            if isinstance(item, list):
                flat_list.extend(item)
            elif item is not None:
                flat_list.append(item)
        return flat_list
