import _lzma
import io
from typing import IO, BinaryIO, List, Optional, Union

import py7zr
from py7zr.io import BytesIOFactory

from ..base import ArchiveHandler
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password

        try:
            with self._open_7z_file() as archive:
                self.filenames = archive.getnames()
                if self.password:
                    # Try to extract the first file to check password
                    factory = BytesIOFactory(limit=1)
                    archive.extract(targets=[self.filenames[0]], factory=factory)

        except _lzma.LZMAError as e:
            raise InvalidPasswordError("Invalid password for 7z file.") from e
        except py7zr.Bad7zFile as e:
            raise InvalidPasswordError("Invalid password for 7z file.") from e

    def _open_7z_file(self) -> py7zr.SevenZipFile:
        """Helper to open the 7z file, handling seeks for file-like objects."""
        if isinstance(self.file, str):
            return py7zr.SevenZipFile(self.file, "r", password=self.password)
        else:
            if hasattr(self.file, "seek"):
                self.file.seek(0)
            return py7zr.SevenZipFile(self.file, "r", password=self.password)

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def list_files(self) -> List[str]:
        return self.filenames

    def open_file(self, file_name: str) -> IO[bytes]:
        try:
            # We need to re-open the file for each extraction.
            with self._open_7z_file() as archive:
                # Use a large limit to avoid issues with large files.
                factory = BytesIOFactory(limit=1024 * 1024 * 1024)  # 1GB limit
                archive.extract(targets=[file_name], factory=factory)
                if file_name in factory.products:
                    product = factory.get(file_name)  # type: ignore
                    if product:
                        product.seek(0)
                        # We need to return a file-like object that supports the context manager protocol
                        # so we wrap the bytes in a BytesIO object.
                        return io.BytesIO(product.read())
                    else:
                        raise IOError(f"File not found in 7z archive: {file_name}")
                else:
                    raise IOError(f"File not found in 7z archive: {file_name}")
        except py7zr.Bad7zFile as e:
            # This could happen for file-specific passwords, though less common.
            raise InvalidPasswordError(f"Invalid password for file '{file_name}' in 7z archive.") from e


def register() -> None:
    register_handler(SevenZipArchiveHandler)


register()
