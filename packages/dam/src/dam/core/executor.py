# pyright: basic
import asyncio
from typing import Any, AsyncGenerator, Generic, List, Optional, TypeVar

from dam.core.enums import ExecutionStrategy
from dam.core.system_events import BaseSystemEvent, SystemResultEvent

ResultType = TypeVar("ResultType")


class SystemExecutor(Generic[ResultType]):
    """
    Executes a collection of system generators according to a specified strategy
    and provides methods to consume the resulting event stream.
    """

    def __init__(
        self,
        generators: List[AsyncGenerator[BaseSystemEvent, None]],
        strategy: ExecutionStrategy,
    ):
        self._generators = generators
        self._strategy = strategy
        self._results: Optional[List[ResultType]] = None
        self._iterator: Optional[AsyncGenerator[BaseSystemEvent, None]] = None

    def __aiter__(self) -> AsyncGenerator[BaseSystemEvent, None]:
        # To prevent re-running the generators, we create the iterator once and reuse it.
        if self._iterator is None:
            if self._strategy == ExecutionStrategy.SERIAL:
                self._iterator = self._run_serial()
            elif self._strategy == ExecutionStrategy.PARALLEL:
                self._iterator = self._run_parallel()
            else:
                raise ValueError(f"Unknown execution strategy: {self._strategy}")
        return self._iterator

    async def _run_serial(self) -> AsyncGenerator[BaseSystemEvent, None]:
        """Executes generators one by one."""
        for gen in self._generators:
            async for event in gen:
                yield event

    async def _run_parallel(self) -> AsyncGenerator[BaseSystemEvent, None]:
        """Executes generators concurrently, yielding events as they become available."""
        q: asyncio.Queue = asyncio.Queue()

        async def drain(gen: AsyncGenerator[BaseSystemEvent, None]):
            """Drains a generator's items into a queue."""
            try:
                async for item in gen:
                    await q.put(item)
            except Exception as e:
                # If a generator fails, put the exception in the queue to be re-raised.
                await q.put(e)

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
                if isinstance(event, SystemResultEvent):
                    self._results.append(event.result)  # type: ignore

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

    async def get_all_results_flat(self) -> List[Any]:
        """
        Consumes the stream, gets all results, and flattens any that are lists.
        """
        results = await self.get_all_results()
        flat_list = []
        for item in results:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list
