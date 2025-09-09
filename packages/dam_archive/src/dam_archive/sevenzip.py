from typing import IO, Any, List, Optional

import py7zr

from .base import ArchiveHandler
from .py7zio import InMemoryIOFactory
from .registry import register_handler


class SevenZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for 7z files.
    """

    def __init__(self, file: Any, passwords: Optional[List[str]] = None):
        self.file = file
        self.passwords = passwords

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".7z")

    def _open_archive(self, pwd: Optional[str] = None) -> py7zr.SevenZipFile:
        if hasattr(self.file, "seek"):
            self.file.seek(0)
        return py7zr.SevenZipFile(self.file, "r", password=pwd)

    def _try_open(self) -> py7zr.SevenZipFile:
        if self.passwords:
            for pwd in self.passwords:
                try:
                    return self._open_archive(pwd)
                except py7zr.Bad7zFile:
                    continue
        return self._open_archive()

    def list_files(self) -> List[str]:
        with self._try_open() as archive:
            return archive.getnames()

    def open_file(self, file_name: str) -> IO[bytes]:
        with self._try_open() as archive:
            factory = InMemoryIOFactory()
            archive.extract(targets=[file_name], factory=factory)
            if file_name in factory.products:
                return factory.products[file_name].get_bytes_io()
            else:
                raise IOError(f"File not found in 7z archive: {file_name}")


def register() -> None:
    register_handler(SevenZipArchiveHandler)
