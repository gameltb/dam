from typing import IO, BinaryIO, List, Optional, Union

import py7zr

from ..base import ArchiveHandler
from ..exceptions import InvalidPasswordError
from ..py7zio import InMemoryIOFactory
from ..registry import register_handler


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password

        try:
            if isinstance(self.file, str):
                self.sevenzip_file = py7zr.SevenZipFile(self.file, "r", password=self.password)
            else:
                if hasattr(self.file, "seek"):
                    self.file.seek(0)
                self.sevenzip_file = py7zr.SevenZipFile(self.file, "r", password=self.password)

            if self.password:
                # Try to extract the first file to check password
                factory = InMemoryIOFactory()
                self.sevenzip_file.extract(targets=[self.sevenzip_file.getnames()[0]], factory=factory)

        except py7zr.Bad7zFile as e:
            raise InvalidPasswordError("Invalid password for 7z file.") from e

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def list_files(self) -> List[str]:
        return self.sevenzip_file.getnames()

    def open_file(self, file_name: str) -> IO[bytes]:
        try:
            factory = InMemoryIOFactory()
            self.sevenzip_file.extract(targets=[file_name], factory=factory)
            if file_name in factory.products:
                return factory.products[file_name].get_bytes_io()
            else:
                raise IOError(f"File not found in 7z archive: {file_name}")
        except py7zr.Bad7zFile as e:
            # This could happen for file-specific passwords, though less common.
            raise InvalidPasswordError(f"Invalid password for file '{file_name}' in 7z archive.") from e


def register() -> None:
    register_handler(SevenZipArchiveHandler)

register()
