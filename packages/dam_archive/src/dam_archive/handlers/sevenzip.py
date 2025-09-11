import logging
import lzma
import os
import threading
from pathlib import PurePosixPath
from queue import Empty, Full, Queue
from typing import IO, BinaryIO, Callable, Dict, Iterable, Iterator, List, Optional, Union

import py7zr
from py7zr.exceptions import PasswordRequired
from py7zr.io import Py7zIO, WriterFactory

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler

logger = logging.getLogger(__name__)


class _StreamingManager:
    def __init__(self, thread: threading.Thread, count: int):
        self._thread = thread
        self._lock = threading.Lock()
        self._count = count

    def file_closed(self) -> None:
        with self._lock:
            self._count -= 1
            if self._count == 0:
                if self._thread.is_alive():
                    self._thread.join()


class SevenZipArchiveFile(ArchiveFile):
    """
    Represents a file within a 7z archive, backed by a streaming pipe.
    """

    def __init__(self, member_info: ArchiveMemberInfo, stream_factory: Callable[[], IO[bytes]]):
        self._member_info = member_info
        self._stream_factory = stream_factory
        self._stream: Optional[IO[bytes]] = None
        self._closed = False
        self.manager: Optional[_StreamingManager] = None

    def _get_stream(self) -> IO[bytes]:
        if self._stream is None:
            self._stream = self._stream_factory()
        return self._stream

    @property
    def name(self) -> str:
        return self._member_info.name

    @property
    def size(self) -> int:
        return self._member_info.size

    def read(self, n: int = -1) -> bytes:
        return self._get_stream().read(n)

    def close(self) -> None:
        if not self._closed:
            if self._stream:
                # Drain the pipe to unblock the producer thread
                while self._stream.read(1024 * 8):
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
        self._closed = True
        if self.manager:
            self.manager.file_closed()

    def __enter__(self) -> "SevenZipArchiveFile":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def readline(self, limit: int = -1) -> bytes:
        return self._get_stream().readline(limit)

    def readlines(self, hint: int = -1) -> list[bytes]:
        return self._get_stream().readlines(hint)

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        raise OSError("seek is not supported on this file-like object")

    def tell(self) -> int:
        raise OSError("tell is not supported on this file-like object")

    def fileno(self) -> int:
        return self._get_stream().fileno()

    def flush(self) -> None:
        self._get_stream().flush()

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


class StreamingPy7zIO(Py7zIO):
    """A file-like object that writes to a pipe."""

    def __init__(self, write_pipe: int):
        self._pipe = write_pipe
        self._size = 0

    def write(self, s: Union[bytes, bytearray]) -> int:
        # write returns number of bytes written
        n = os.write(self._pipe, s)
        try:
            self._size += n
        except Exception:
            pass
        return n

    def read(self, size: Optional[int] = None) -> bytes:
        # writer IO doesn't support read; return empty bytes
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        raise OSError("seek not supported")

    def flush(self) -> None:
        return None

    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        os.close(self._pipe)


class StreamingFactory(WriterFactory):
    """
    A factory that creates a pipe for each file and puts the readable
    end on a queue for the consumer.
    """

    def __init__(
        self,
        queue: "Queue[SevenZipArchiveFile | Exception | None]",
        members: List[ArchiveMemberInfo],
        handler: "SevenZipArchiveHandler",
    ):
        self.queue: "Queue[SevenZipArchiveFile | Exception | None]" = queue
        self.handler = handler
        # Normalize to POSIX paths to match py7zr's filename format
        self.members_map = {PurePosixPath(m.name).as_posix(): m for m in members}

    def create(self, filename: str, mode: str = "wb") -> Optional[Py7zIO]:  # type: ignore[override]
        norm = PurePosixPath(filename).as_posix()
        if norm in self.members_map:
            read_pipe, write_pipe = os.pipe()
            member_info = self.members_map[norm]

            def stream_factory() -> IO[bytes]:
                return os.fdopen(read_pipe, "rb")

            archive_file = SevenZipArchiveFile(member_info, stream_factory)
            self.queue.put(archive_file)
            return StreamingPy7zIO(write_pipe)
        return None


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files that supports true streaming extraction.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        self.members: List[ArchiveMemberInfo] = []
        self._streaming_state: Optional[
            Dict[str, Union[threading.Thread, Queue["SevenZipArchiveFile | Exception | None"], List[ArchiveFile]]]
        ] = None

        try:
            with self._open_7z_file() as archive:
                for member in archive.list():  # type: ignore
                    if not member.is_directory:
                        self.members.append(ArchiveMemberInfo(name=member.filename, size=member.uncompressed))  # type: ignore
        except (lzma.LZMAError, py7zr.Bad7zFile, PasswordRequired) as e:
            # py7zr may raise lzma errors, Bad7zFile for corrupted archives,
            # or PasswordRequired when a password is needed/incorrect. Map
            # these to our InvalidPasswordError to match tests and caller
            # expectations while preserving the original exception as context.
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    def _streaming_iter_files(self) -> Iterator[ArchiveFile]:
        q: "Queue[SevenZipArchiveFile | Exception | None]" = Queue()

        def producer() -> None:
            try:
                factory = StreamingFactory(q, self.members, self)
                with self._open_7z_file() as archive:
                    archive.extractall(factory=factory)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        thread = threading.Thread(target=producer)
        thread.start()
        manager = _StreamingManager(thread, len(self.members))
        processed_files = 0
        try:
            while processed_files < len(self.members):
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                item.manager = manager
                yield item
                processed_files += 1
        except GeneratorExit:
            # The generator was closed early, ensure we clean up
            manager.file_closed()
            raise

    def _streaming_open_file(self, file_name: str) -> ArchiveFile:
        q: "Queue[SevenZipArchiveFile | Exception | None]" = Queue(1)

        def producer() -> None:
            try:
                factory = StreamingFactory(q, self.members, self)
                with self._open_7z_file() as archive:
                    archive.extract(targets=[file_name], factory=factory)
            except Exception as e:
                q.put(e)
            finally:
                try:
                    q.put_nowait(None)
                except Full:
                    pass

        thread = threading.Thread(target=producer)
        thread.start()
        manager = _StreamingManager(thread, 1)

        try:
            item = q.get(timeout=5.0)
            if isinstance(item, Exception):
                raise item
            if item is None:
                raise IOError(f"File not found in 7z archive: {file_name}")
            item.manager = manager
            return item
        except Empty:
            raise IOError(f"Timeout waiting for file stream: {file_name}")

    def close(self) -> None:
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

    def iter_files(self) -> Iterator[ArchiveFile]:
        return self._streaming_iter_files()

    def open_file(self, file_name: str) -> ArchiveFile:
        return self._streaming_open_file(file_name)


def register() -> None:
    register_handler(SevenZipArchiveHandler)


register()
