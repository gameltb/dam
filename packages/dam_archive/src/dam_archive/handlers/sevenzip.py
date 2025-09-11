import _lzma
import logging
import os
import threading
from pathlib import PurePosixPath
from queue import Queue
from typing import IO, BinaryIO, Iterator, List, Optional, Union

import py7zr
from py7zr.io import Py7zIO, WriterFactory

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler

logger = logging.getLogger(__name__)


class SevenZipArchiveFile(ArchiveFile):
    """
    Represents a file within a 7z archive, backed by a streaming pipe.
    """

    def __init__(self, member_info: ArchiveMemberInfo, stream: IO[bytes]):
        self._member_info = member_info
        self._stream = stream

    @property
    def name(self) -> str:
        return self._member_info.name

    @property
    def size(self) -> int:
        return self._member_info.size

    def open(self) -> IO[bytes]:
        return self._stream


class StreamingPy7zIO(Py7zIO):
    """A file-like object that writes to a pipe."""

    def __init__(self, write_pipe: int):
        self._pipe = write_pipe

    def write(self, data: bytes) -> int:
        return os.write(self._pipe, data)

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        os.close(self._pipe)


class StreamingFactory(WriterFactory):
    """
    A factory that creates a pipe for each file and puts the readable
    end on a queue for the consumer.
    """

    def __init__(self, queue: Queue, members: List[ArchiveMemberInfo]):
        self.queue = queue
        self.members_map = {m.name: m for m in members}

    def create(self, filename: str, mode: str = "wb") -> Optional[StreamingPy7zIO]:
        if filename in self.members_map:
            read_pipe, write_pipe = os.pipe()
            member_info = self.members_map[filename]
            read_stream = os.fdopen(read_pipe, "rb")
            archive_file = SevenZipArchiveFile(member_info, read_stream)
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

        try:
            with self._open_7z_file() as archive:
                for member in archive.list():
                    if not member.is_directory:
                        self.members.append(
                            ArchiveMemberInfo(name=member.filename, size=member.uncompressed)
                        )
        except (_lzma.LZMAError, py7zr.Bad7zFile) as e:
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    def close(self) -> None:
        pass

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[ArchiveFile]:
        q: Queue = Queue()

        def producer():
            try:
                factory = StreamingFactory(q, self.members)
                with self._open_7z_file() as archive:
                    archive.extractall(factory=factory)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        thread = threading.Thread(target=producer)
        thread.start()

        processed_files = 0
        while processed_files < len(self.members):
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            processed_files += 1
            yield item

        thread.join()


    def open_file(self, file_name: str) -> IO[bytes]:
        q: Queue = Queue()

        def producer():
            try:
                factory = StreamingFactory(q, self.members)
                with self._open_7z_file() as archive:
                    archive.extract(targets=[file_name], factory=factory)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        thread = threading.Thread(target=producer)
        thread.start()

        item = q.get()
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise IOError(f"File not found in 7z archive: {file_name}")

        thread.join()
        return item.open()


def register() -> None:
    register_handler(SevenZipArchiveHandler)


register()
