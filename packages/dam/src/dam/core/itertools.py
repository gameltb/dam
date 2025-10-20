"""Core itertools for the DAM system."""

from collections.abc import AsyncGenerator


async def asend_wrapper[T, V](
    agen: AsyncGenerator[T, V],
) -> AsyncGenerator[T, V]:
    """Wrap an async generator to simplify the asend pattern."""
    try:
        item = await anext(agen)
        while True:
            sent_val = yield item
            item = await agen.asend(sent_val)
    except StopAsyncIteration:
        pass
