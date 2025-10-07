"""Core type definitions for the DAM system."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO


class StreamProvider(ABC):
    """
    An abstract base class for providing data streams.

    This class defines a common interface for accessing data as either a binary
    stream (`BinaryIO`) or a file path (`Path`).
    """

    @abstractmethod
    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        """
        Provide a fresh, readable binary stream in an async context.

        The stream is automatically closed upon exiting the context.
        """
        if False:
            yield

    def get_path(self) -> Path | None:
        """
        Return the path to the file, if available.

        This is a hint for optimization. If this returns None, the caller should
        fall back to using get_stream().
        """
        return None


class CallableStreamProvider(StreamProvider):
    """A stream provider that wraps a callable."""

    def __init__(self, stream_factory: Callable[[], BinaryIO]):
        """Initialize the provider with a stream factory."""
        self._stream_factory = stream_factory

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        """Get a stream from the factory and ensure it's closed."""
        stream = self._stream_factory()
        try:
            yield stream
        finally:
            stream.close()


class FileStreamProvider(StreamProvider):
    """A stream provider that wraps a file path."""

    def __init__(self, path: Path):
        """Initialize the provider with a file path."""
        self._path = path

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        """Open the file path as a binary stream."""
        with self._path.open("rb") as stream:
            yield stream

    def get_path(self) -> Path | None:
        """Return the path to the file."""
        return self._path
