"""Provides a 7z archive handler using py7zr library."""

from __future__ import annotations

import asyncio
import datetime
import logging
import lzma
import threading
import types
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from queue import Queue
from typing import BinaryIO, Self, cast

import py7zr
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.io import Py7zIO, WriterFactory

from ..base import ArchiveHandler, ArchiveMemberInfo, StreamProvider
from ..exceptions import InvalidPasswordError, UnsupportedArchiveError

logger = logging.getLogger(__name__)

QUEUE_MAX_SIZE = 10


class _SevenZipStreamReader(BinaryIO):
    """A file-like object that reads from a streaming queue."""

    def __init__(self, queue: Queue[bytes | Exception | None]):
        self._queue = queue
        self._buffer = b""
        self._closed = False
        self._eof = False
        self._has_read = False

    def read(self, n: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if self._eof and not self._buffer:
            return b""
        self._has_read = True

        if n < 0:
            while not self._eof:
                self._read_from_queue()
            data = self._buffer
            self._buffer = b""
            return data

        while len(self._buffer) < n and not self._eof:
            self._read_from_queue()

        data = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return data

    def _read_from_queue(self) -> None:
        if self._eof:
            return
        item = self._queue.get()
        if isinstance(item, bytes):
            self._buffer += item
        elif isinstance(item, Exception):
            self._eof = True
            raise item
        elif item is None:
            self._eof = True
        else:
            self._eof = True
            raise TypeError(f"Unexpected item in queue: {type(item)}")

    def close(self) -> None:
        if not self._closed:
            while not self._eof:
                try:
                    self._read_from_queue()
                except Exception:
                    break
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def readline(self, limit: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        newline_pos = -1
        while newline_pos == -1 and not self._eof:
            newline_pos = self._buffer.find(b"\n")
            if newline_pos == -1:
                self._read_from_queue()

        if newline_pos != -1:
            if limit != -1 and newline_pos + 1 > limit:
                line = self._buffer[:limit]
                self._buffer = self._buffer[limit:]
            else:
                line = self._buffer[: newline_pos + 1]
                self._buffer = self._buffer[newline_pos + 1 :]
        elif limit != -1:
            line = self._buffer[:limit]
            self._buffer = self._buffer[limit:]
        else:
            line = self._buffer
            self._buffer = b""
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines: list[bytes] = []
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            if hint > 0 and sum(map(len, lines)) >= hint:
                break
        return lines

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        if not self._has_read and offset == 0 and whence in (0, 1):
            return 0
        raise OSError("seek is not supported on this file-like object")

    def tell(self) -> int:
        raise OSError("tell is not supported on this file-like object")

    def fileno(self) -> int:
        raise OSError("fileno is not supported")

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def write(self, _: bytes) -> int:  # type: ignore[override]
        raise OSError("write is not supported")

    def writelines(self, _: Iterable[bytes]) -> None:  # type: ignore[override]
        raise OSError("writelines is not supported")

    def truncate(self, _: int | None = None) -> int:
        raise OSError("truncate is not supported")

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


class QueueWriter(Py7zIO):
    """A file-like object that writes to a queue with buffering."""

    def __init__(self, queue: Queue[bytes | Exception | None]):
        """Initialize the queue writer."""
        self.queue = queue
        self._size = 0
        self._buffer: list[bytes] = []
        self._buffer_size = 0
        self.BUFFER_LIMIT = 100 * 1024 * 1024  # 100MB

    def _flush_buffer(self):
        if not self._buffer:
            return

        # In Python, joining an empty list of bytes with a non-empty separator
        # results in an empty bytes object. However, if the list is truly empty
        # (no chunks were ever added), we should not put an empty bytes object
        # into the queue unless we intend to signal something specific by it.
        # Here, we ensure we only join and put if there's content.
        if self._buffer_size > 0:
            self.queue.put(b"".join(self._buffer))

        self._buffer = []
        self._buffer_size = 0

    def write(self, s: bytes | bytearray) -> int:
        """Write bytes to the buffer."""
        chunk = bytes(s)
        self._buffer.append(chunk)
        self._buffer_size += len(chunk)
        self._size += len(chunk)

        if self._buffer_size > self.BUFFER_LIMIT:
            self._flush_buffer()

        return len(chunk)

    def read(self, size: int | None = None) -> bytes:  # noqa: ARG002
        """Read is not supported."""
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek is not supported."""
        if offset == 0 and whence in (0, 1):
            return 0
        raise OSError(f"seek not supported (offset={offset}, whence={whence})")

    def flush(self) -> None:
        """Flush the buffer to the queue."""
        self._flush_buffer()

    def size(self) -> int:
        """Return the total size of bytes written."""
        return self._size

    def seekable(self) -> bool:
        """Return True if the stream supports random access."""
        return True

    def close(self) -> None:
        """Close the writer and signal end of stream."""
        self._flush_buffer()
        self.queue.put(None)


class StreamingFactory(WriterFactory):
    """A factory that creates a data queue for each file and reports it back to the main thread."""

    def __init__(
        self,
        members_map: dict[str, ArchiveMemberInfo],
        results_queue: Queue[tuple[str, Queue[bytes | Exception | None]] | Exception | None],
    ):
        """Initialize the streaming factory."""
        self.members_map = members_map
        self.results_queue = results_queue
        self.last_writer: QueueWriter | None = None

    def create(self, filename: str, _: str = "wb") -> Py7zIO | None:  # type: ignore[override]
        """Create a writer for a given filename."""
        if self.last_writer:
            self.last_writer.close()
            self.last_writer = None

        norm = PurePosixPath(filename).as_posix()
        if norm in self.members_map:
            data_queue: Queue[bytes | Exception | None] = Queue(maxsize=QUEUE_MAX_SIZE)
            self.results_queue.put((norm, data_queue))
            writer = QueueWriter(data_queue)
            self.last_writer = writer
            return writer
        return None

    def close_last_writer(self) -> None:
        """Close the last writer created by the factory."""
        if self.last_writer:
            self.last_writer.close()
            self.last_writer = None


class SevenZipArchiveHandler(ArchiveHandler):
    """An archive handler for 7z files that supports true streaming extraction."""

    def __init__(self, stream_provider: StreamProvider, password: str | None = None):
        """Initialize the 7z archive handler."""
        super().__init__(stream_provider, password)
        self.members: list[ArchiveMemberInfo] = []
        self._threads: list[threading.Thread] = []

    @classmethod
    async def create(cls, stream_provider: StreamProvider, password: str | None = None) -> SevenZipArchiveHandler:
        """Asynchronously create and initialize a 7z archive handler."""
        handler = cls(stream_provider, password)
        try:
            async with stream_provider.get_stream() as stream:
                with py7zr.SevenZipFile(stream, "r", password=password) as archive:
                    for member in archive.list():
                        if not member.is_directory:
                            handler.members.append(
                                ArchiveMemberInfo(
                                    name=cast(str, member.filename),
                                    size=cast(int, member.uncompressed),
                                    modified_at=cast(datetime.datetime, member.creationtime),
                                    compressed_size=cast(int, member.compressed),
                                )
                            )
                    if handler.members:
                        # testzip() will raise UnsupportedCompressionMethodError if any file
                        # uses an unsupported filter.
                        archive.testzip()

        except UnsupportedCompressionMethodError as e:
            raise UnsupportedArchiveError(f"Unsupported 7z archive: {e}") from e
        except (lzma.LZMAError, py7zr.Bad7zFile, PasswordRequired) as e:
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

        return handler

    def _start_producer(
        self, targets: list[str] | None = None
    ) -> Queue[tuple[str, Queue[bytes | Exception | None]] | Exception | None]:
        results_queue: Queue[tuple[str, Queue[bytes | Exception | None]] | Exception | None] = Queue()
        members_map = {PurePosixPath(m.name).as_posix(): m for m in self.members}

        def producer_thread_target():
            # py7zr is blocking, so we need a separate thread.
            # And because get_stream is async, we need an event loop in this new thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def async_producer():
                factory = StreamingFactory(members_map, results_queue)
                try:
                    async with self._stream_provider.get_stream() as stream:
                        with py7zr.SevenZipFile(stream, "r", password=self.password) as archive:
                            if targets:
                                archive.extract(targets=targets, factory=factory)
                            else:
                                archive.extractall(factory=factory)
                except Exception as e:
                    results_queue.put(e)
                finally:
                    factory.close_last_writer()
                    results_queue.put(None)

            loop.run_until_complete(async_producer())
            loop.close()

        thread = threading.Thread(target=producer_thread_target)
        thread.start()
        self._threads.append(thread)
        return results_queue

    def iter_files(self) -> Iterator[tuple[ArchiveMemberInfo, BinaryIO]]:
        """Iterate over all files in the archive."""
        results_queue = self._start_producer()
        while True:
            item = results_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            norm_name, data_queue = item
            member_info = next((m for m in self.members if PurePosixPath(m.name).as_posix() == norm_name), None)
            if member_info:
                yield member_info, _SevenZipStreamReader(data_queue)

    def open_file(self, file_name: str) -> tuple[ArchiveMemberInfo, BinaryIO]:
        """Open a specific file from the archive."""
        norm_name = PurePosixPath(file_name).as_posix()
        member_info = next((m for m in self.members if PurePosixPath(m.name).as_posix() == norm_name), None)
        if not member_info:
            raise OSError(f"File not found in 7z archive: {file_name}")

        results_queue = self._start_producer(targets=[file_name])

        while True:
            item = results_queue.get()
            if item is None:
                raise OSError(f"File not found in 7z archive stream: {file_name}")
            if isinstance(item, Exception):
                raise item

            current_norm_name, data_queue = item
            if current_norm_name == norm_name:
                return member_info, _SevenZipStreamReader(data_queue)
            # This is a stream for a different file, which can happen
            # if py7zr decides to extract dependencies. We need to drain it.
            temp_file = _SevenZipStreamReader(data_queue)
            temp_file.close()

    async def close(self) -> None:
        """Close the archive and join any running threads."""
        for t in self._threads:
            t.join()

    def list_files(self) -> list[ArchiveMemberInfo]:
        """List all files in the archive."""
        return self.members
