import _lzma
import io
from typing import IO, BinaryIO, Iterator, List, Optional, Union

import py7zr
from py7zr.io import BytesIOFactory

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


class SevenZipArchiveFile(ArchiveFile):
    """
    Represents a file within a 7z archive, backed by an in-memory stream.
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
        self._stream.seek(0)
        return self._stream


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        self.members: List[ArchiveMemberInfo] = []
        self.filenames: List[str] = []

        try:
            with self._open_7z_file() as archive:
                for member in archive.list():
                    if not member.filename.endswith("/"):  # type: ignore
                        self.members.append(
                            ArchiveMemberInfo(name=member.filename, size=member.uncompressed)  # type: ignore
                        )
                        self.filenames.append(member.filename)  # type: ignore

                if self.password and self.filenames:
                    # Try to extract the first file to check password
                    factory = BytesIOFactory(limit=1)
                    archive.extract(targets=[self.filenames[0]], factory=factory)

        except (_lzma.LZMAError, py7zr.Bad7zFile) as e:
            raise InvalidPasswordError("Invalid password or corrupted 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        """Helper to open the 7z file, handling seeks for file-like objects."""
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    def close(self) -> None:
        """Does nothing, as the archive is opened on demand."""
        pass

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[ArchiveFile]:
        """
        Iterate over all files in the archive.

        Note: This implementation uses `extractall()`, which extracts the entire
        archive into memory. This is a trade-off for performance and to work
        around issues in the py7zr library, but it can consume a large
        amount of memory for large archives.
        """
        factory = BytesIOFactory(limit=1024 * 1024 * 1024)  # 1GB limit
        with self._open_7z_file() as archive:
            archive.extractall(factory=factory)

        for member_info in self.members:
            if member_info.name in factory.products:
                stream = factory.get(member_info.name)  # type: ignore
                if stream:
                    yield SevenZipArchiveFile(member_info, io.BytesIO(stream.read()))

    def open_file(self, file_name: str) -> IO[bytes]:
        """
        Opens a single file from the archive.

        Note: This implementation uses `extract()`, which extracts the requested
        file into memory.
        """
        # Note: We re-open the archive for each file extraction.
        # This is a workaround for a defect in py7zr where multiple `extract`
        # calls on the same SevenZipFile instance can lead to a deadlock.
        # Re-opening the archive for each file is less efficient but more stable.
        try:
            factory = BytesIOFactory(limit=1024 * 1024 * 1024)  # 1GB limit
            with self._open_7z_file() as archive:
                archive.extract(targets=[file_name], factory=factory)
                if file_name in factory.products:
                    stream = factory.get(file_name)  # type: ignore
                    if stream:
                        return io.BytesIO(stream.read())
        except (_lzma.LZMAError, py7zr.Bad7zFile) as e:
            raise InvalidPasswordError(f"Invalid password for file '{file_name}' in 7z archive.") from e

        raise IOError(f"File not found in 7z archive: {file_name}")


def register() -> None:
    register_handler(SevenZipArchiveHandler)


register()
