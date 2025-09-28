from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Callable, Optional


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
        An async context manager that provides a fresh, readable binary stream.
        The stream is automatically closed upon exiting the context.
        """
        if False:
            yield

    def get_path(self) -> Optional[Path]:
        """
        Returns the path to the file, if available.

        This is a hint for optimization. If this returns None, the caller should
        fall back to using get_stream().
        """
        return None


class CallableStreamProvider(StreamProvider):
    """
    A stream provider that wraps a callable.
    """

    def __init__(self, stream_factory: Callable[[], BinaryIO]):
        self._stream_factory = stream_factory

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        stream = self._stream_factory()
        try:
            yield stream
        finally:
            stream.close()


class FileStreamProvider(StreamProvider):
    """
    A stream provider that wraps a file path.
    """

    def __init__(self, path: Path):
        self._path = path

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        with self._path.open("rb") as stream:
            yield stream

    def get_path(self) -> Optional[Path]:
        return self._path
