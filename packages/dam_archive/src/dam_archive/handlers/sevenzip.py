import logging
import lzma
import threading
from pathlib import PurePosixPath
from queue import Queue
from typing import IO, BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import py7zr
from py7zr.exceptions import PasswordRequired
from py7zr.io import Py7zIO, WriterFactory

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler

logger = logging.getLogger(__name__)

QUEUE_MAX_SIZE = 100


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

    def __enter__(self) -> "SevenZipArchiveFile":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
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
        if offset == 0 and whence in (0, 1):
            return 0
        raise OSError(f"seek not supported (offset={offset}, whence={whence})")

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.queue.put(None)


class StreamingFactory(WriterFactory):
    """
    A factory that creates a data queue for each file and reports it back to the main thread.
    """

    def __init__(
        self,
        members_map: Dict[str, ArchiveMemberInfo],
        results_queue: "Queue[Union[Tuple[str, Queue[Union[bytes, Exception, None]]], Exception, None]]",
    ):
        self.members_map = members_map
        self.results_queue = results_queue
        self.last_writer: Optional[QueueWriter] = None

    def create(self, filename: str, mode: str = "wb") -> Optional[Py7zIO]:  # type: ignore[override]
        if self.last_writer:
            self.last_writer.close()
            self.last_writer = None

        norm = PurePosixPath(filename).as_posix()
        if norm in self.members_map:
            data_queue: "Queue[Union[bytes, Exception, None]]" = Queue(maxsize=QUEUE_MAX_SIZE)
            self.results_queue.put((norm, data_queue))
            writer = QueueWriter(data_queue)
            self.last_writer = writer
            return writer
        return None

    def close_last_writer(self) -> None:
        if self.last_writer:
            self.last_writer.close()
            self.last_writer = None


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
                if self.password and self.members:
                    with self._open_7z_file() as test_archive:
                        test_archive.testzip()
                    if hasattr(self.file, "seek"):
                        self.file.seek(0)  # type: ignore

        except (lzma.LZMAError, py7zr.Bad7zFile, PasswordRequired) as e:
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    def _start_producer(
        self, targets: Optional[List[str]] = None
    ) -> "Queue[Union[Tuple[str, Queue[Union[bytes, Exception, None]]], Exception, None]]":
        results_queue: "Queue[Union[Tuple[str, Queue[Union[bytes, Exception, None]]], Exception, None]]" = Queue()
        members_map = {PurePosixPath(m.name).as_posix(): m for m in self.members}

        def producer() -> None:
            factory = StreamingFactory(members_map, results_queue)
            try:
                with self._open_7z_file() as archive:
                    if targets:
                        archive.extract(targets=targets, factory=factory)
                    else:
                        archive.extractall(factory=factory)
            except Exception as e:
                results_queue.put(e)
            finally:
                factory.close_last_writer()
                results_queue.put(None)

        thread = threading.Thread(target=producer)
        thread.start()
        self._threads.append(thread)
        return results_queue

    def iter_files(self) -> Iterator[ArchiveFile]:
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
                yield SevenZipArchiveFile(member_info, data_queue)

    def open_file(self, file_name: str) -> ArchiveFile:
        norm_name = PurePosixPath(file_name).as_posix()
        member_info = next((m for m in self.members if PurePosixPath(m.name).as_posix() == norm_name), None)
        if not member_info:
            raise IOError(f"File not found in 7z archive: {file_name}")

        results_queue = self._start_producer(targets=[file_name])

        while True:
            item = results_queue.get()
            if item is None:
                raise IOError(f"File not found in 7z archive stream: {file_name}")
            if isinstance(item, Exception):
                raise item

            current_norm_name, data_queue = item
            if current_norm_name == norm_name:
                return SevenZipArchiveFile(member_info, data_queue)
            else:
                # This is a stream for a different file, which can happen
                # if py7zr decides to extract dependencies. We need to drain it.
                temp_file = SevenZipArchiveFile(member_info, data_queue)
                temp_file.close()

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
