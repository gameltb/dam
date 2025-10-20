"""Tests for the core itertools."""

from collections.abc import AsyncGenerator

import pytest

from dam.core.itertools import asend_wrapper


async def simple_async_gen() -> AsyncGenerator[int, str | None]:
    """Provide a simple async generator for testing."""
    value: str | None = None
    count = 0
    while count < 3:
        value = yield count
        count += 1 if value is None else int(value)


@pytest.mark.asyncio
async def test_asend_wrapper_basic():
    """Test basic functionality of asend_wrapper."""
    gen = asend_wrapper(simple_async_gen())
    results: list[int] = []
    async for item in gen:
        results.append(item)
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_asend_wrapper_with_send():
    """Test asend_wrapper with sending values."""
    gen = asend_wrapper(simple_async_gen())
    assert await anext(gen) == 0
    assert await gen.asend("2") == 2
    with pytest.raises(StopAsyncIteration):
        await anext(gen)
