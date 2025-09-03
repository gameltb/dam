import zipfile
from typing import IO, List, Optional

from .base import ArchiveHandler
from .registry import register_handler


class ZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for zip files.
    """

    def __init__(self, file_path: str, password: Optional[str] = None):
        self.file_path = file_path
        self.password = password
        try:
            self.zip_file = zipfile.ZipFile(self.file_path, "r")
            if self.password:
                self.zip_file.setpassword(self.password.encode())
        except (RuntimeError, zipfile.BadZipFile) as e:
            raise IOError(f"Failed to open zip file: {e}") from e

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".zip")

    def list_files(self) -> List[str]:
        return self.zip_file.namelist()

    def open_file(self, file_name: str) -> IO[bytes]:
        try:
            return self.zip_file.open(file_name)
        except (RuntimeError, KeyError) as e:
            raise IOError(f"Failed to open file in zip: {e}") from e


def register():
    register_handler(ZipArchiveHandler)


register()
