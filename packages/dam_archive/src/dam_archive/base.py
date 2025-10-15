"""Base classes and utilities for archive handling."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from dam.core.types import CallableStreamProvider, FileStreamProvider, StreamProvider


def to_stream_provider(
    file_or_path: str | BinaryIO | Path,
) -> StreamProvider:
    """
    Convert a file path or a binary stream into a StreamProvider.

    Args:
        file_or_path: The file path, Path object, or binary stream.

    Returns:
        A StreamProvider instance.

    Raises:
        ValueError: If the stream is not seekable.

    """
    if isinstance(file_or_path, str):
        return FileStreamProvider(Path(file_or_path))
    if isinstance(file_or_path, Path):
        return FileStreamProvider(file_or_path)
    # If the stream is seekable, we can read it into memory and use it.
    if file_or_path.seekable():
        file_or_path.seek(0)
        # Read the content into an immutable bytes object.
        content = file_or_path.read()

        def bytes_stream_provider() -> BinaryIO:
            # Create a new stream from the bytes object each time.
            return io.BytesIO(content)

        return CallableStreamProvider(bytes_stream_provider)
    # If the stream is not seekable, we cannot create a reliable
    # stream provider from it.
    raise ValueError("Cannot create a stream provider from a non-seekable stream.")


@dataclass
class ArchiveMemberInfo:
    """Represents information about a member in an archive."""

    name: str
    size: int
    modified_at: datetime | None
    compressed_size: int | None = None


class ArchiveHandler(ABC):
    """Abstract base class for archive handlers."""

    def __init__(self, stream_provider: StreamProvider, password: str | None = None):
        """
        Initialize the archive handler.

        Args:
            stream_provider: A StreamProvider instance for the archive.
            password: The password for the archive, if any.

        """
        self._stream_provider = stream_provider
        self.password = password

    @classmethod
    @abstractmethod
    async def create(cls, stream_provider: StreamProvider, password: str | None = None) -> ArchiveHandler:
        """Asynchronously creates and initializes an archive handler."""
        ...

    @property
    def comment(self) -> str | None:
        """The comment of the archive, if any."""
        return None

    @abstractmethod
    def list_files(self) -> list[ArchiveMemberInfo]:
        """List all file names and sizes in the archive."""
        ...

    @abstractmethod
    def iter_files(self) -> Iterator[tuple[ArchiveMemberInfo, BinaryIO]]:
        """
        Iterate over all files in the archive in their natural order.

        This method is designed for efficient, sequential processing of archive
        members. It yields tuples of (`ArchiveMemberInfo`, `BinaryIO`), which can be
        used to access member information and open a stream to the file's content.

        For archive formats that support it, this method should be implemented
        to stream data from the archive rather than performing random-access reads,
        which can be inefficient, especially for solid archives.
        """
        ...

    @abstractmethod
    def open_file(self, file_name: str) -> tuple[ArchiveMemberInfo, BinaryIO]:
        """Open a specific file from the archive and return a file-like object."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close the archive file and release any resources.

        This should be called when the handler is no longer needed.
        """
        ...
