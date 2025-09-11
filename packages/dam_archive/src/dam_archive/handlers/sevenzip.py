import logging
import lzma
import threading
from pathlib import PurePosixPath
from queue import Queue
from typing import IO, BinaryIO, Dict, Iterable, Iterator, List, Optional, Union

import py7zr
from py7zr.exceptions import PasswordRequired
from py7zr.io import Py7zIO, WriterFactory

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler

logger = logging.getLogger(__name__)


class SevenZipArchiveFile(ArchiveFile):
    """
    Represents a file within a 7z archive, backed by a streaming queue.
    """

    def __init__(self, member_info: ArchiveMemberInfo, queue: "Queue[Union[bytes, Exception, None]]"):
        self._member_info = member_info
        self._queue = queue
        self._buffer = b""
        self._closed = False
        self._eof = False

    @property
    def name(self) -> str:
        return self._member_info.name

    @property
    def size(self) -> int:
        return self._member_info.size

    def read(self, n: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if self._eof and not self._buffer:
            return b""

        if n < 0:
            # Read everything
            while not self._eof:
                self._read_from_queue()
            data = self._buffer
            self._buffer = b""
            return data

        # Read until we have enough data or hit EOF
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
            # Should be unreachable
            self._eof = True
            raise TypeError(f"Unexpected item in queue: {type(item)}")

    def close(self) -> None:
        if not self._closed:
            # Consume the rest of the queue to allow the producer to exit
            while not self._eof:
                self._read_from_queue()
            self._closed = True

    def __enter__(self) -> "SevenZipArchiveFile":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.close()

    def readline(self, limit: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        # Read until we find a newline or hit EOF
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
        # Simple, non-optimized implementation
        lines: List[bytes] = []
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

    def write(self, s: bytes) -> int:  # type: ignore[override]
        raise OSError("write is not supported")

    def writelines(self, lines: Iterable[bytes]) -> None:  # type: ignore[override]
        raise OSError("writelines is not supported")

    def truncate(self, size: Optional[int] = None) -> int:
        raise OSError("truncate is not supported")

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


class QueueWriter(Py7zIO):
    """A file-like object that writes to a queue."""

    def __init__(self, queue: "Queue[Union[bytes, Exception, None]]"):
        self.queue = queue
        self._size = 0

    def write(self, s: Union[bytes, bytearray]) -> int:
        chunk = bytes(s)
        self.queue.put(chunk)
        self._size += len(chunk)
        return len(chunk)

    def read(self, size: Optional[int] = None) -> bytes:
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        if offset == 0 and whence in (0, 1):  # os.SEEK_SET, os.SEEK_CUR
            return 0
        raise OSError("seek not supported")

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        # Lie to py7zr to make it happy
        return True

    def close(self) -> None:
        self.queue.put(None)


class StreamingFactory(WriterFactory):
    """
    A factory that creates a queue for each file.
    """

    def __init__(
        self,
        handler: "SevenZipArchiveHandler",
        members_map: Dict[str, ArchiveMemberInfo],
        file_queues: Dict[str, "Queue[Union[bytes, Exception, None]]"],
    ):
        self.handler = handler
        self.members_map = members_map
        self.file_queues = file_queues

    def create(self, filename: str, mode: str = "wb") -> Optional[Py7zIO]:  # type: ignore[override]
        norm = PurePosixPath(filename).as_posix()
        if norm in self.members_map:
            q: "Queue[Union[bytes, Exception, None]]" = Queue()
            self.file_queues[norm] = q
            return QueueWriter(q)
        return None


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files that supports true streaming extraction.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        self.members: List[ArchiveMemberInfo] = []
        self._threads: List[threading.Thread] = []

        try:
            with self._open_7z_file() as archive:
                for member in archive.list():  # type: ignore
                    if not member.is_directory:
                        self.members.append(ArchiveMemberInfo(name=member.filename, size=member.uncompressed))  # type: ignore
        except (lzma.LZMAError, py7zr.Bad7zFile, PasswordRequired) as e:
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    def _start_producer(self, targets: Optional[List[str]] = None) -> Dict[str, "Queue[Union[bytes, Exception, None]]"]:
        file_queues: Dict[str, "Queue[Union[bytes, Exception, None]]"] = {}
        members_map = {PurePosixPath(m.name).as_posix(): m for m in self.members}

        def producer() -> None:
            try:
                factory = StreamingFactory(self, members_map, file_queues)
                with self._open_7z_file() as archive:
                    archive.extract(targets=targets, factory=factory)
            except Exception as e:
                # If extraction fails, put the exception in all queues
                for q in file_queues.values():
                    q.put(e)

        thread = threading.Thread(target=producer)
        thread.start()
        self._threads.append(thread)
        return file_queues

    def iter_files(self) -> Iterator[ArchiveFile]:
        file_queues = self._start_producer()
        for member in self.members:
            norm_name = PurePosixPath(member.name).as_posix()
            q = file_queues.get(norm_name)
            if q:
                yield SevenZipArchiveFile(member, q)

    def open_file(self, file_name: str) -> ArchiveFile:
        norm_name = PurePosixPath(file_name).as_posix()
        member_info = next((m for m in self.members if PurePosixPath(m.name).as_posix() == norm_name), None)
        if not member_info:
            raise IOError(f"File not found in 7z archive: {file_name}")

        file_queues = self._start_producer(targets=[file_name])
        q = file_queues.get(norm_name)
        if not q:
            # This should not happen if the producer works correctly
            raise IOError(f"Failed to create stream for file: {file_name}")
        return SevenZipArchiveFile(member_info, q)

    def close(self) -> None:
        for t in self._threads:
            t.join()
        if isinstance(self.file, IO):
            try:
                self.file.close()
            except Exception:
                pass

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members


def register() -> None:
    register_handler(SevenZipArchiveHandler)


register()
