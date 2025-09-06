import io
import zipfile
from typing import IO, Any, List, Optional

from .base import ArchiveHandler
from .registry import register_handler


class ZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for zip files.
    """

    def __init__(self, file: Any, passwords: Optional[List[str]] = None):
        self.file = file
        self.passwords = passwords
        try:
            self.zip_file = zipfile.ZipFile(self.file, "r")
            if self.passwords:
                self.zip_file.setpassword(self.passwords[0].encode())
                if self.zip_file.testzip() is not None:
                    raise RuntimeError("Bad password for file")
        except (RuntimeError, zipfile.BadZipFile) as e:
            raise IOError(f"Failed to open zip file: {e}") from e

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".zip")

    def list_files(self) -> List[str]:
        return self.zip_file.namelist()

    def open_file(self, file_name: str) -> IO[bytes]:
        path_parts = file_name.split("/", 1)
        if len(path_parts) == 1 or not path_parts[0].lower().endswith(".zip"):
            try:
                return self.zip_file.open(file_name)
            except (RuntimeError, KeyError) as e:
                raise IOError(f"Failed to open file in zip: {e}") from e
        else:
            nested_archive_name, path_in_nested_archive = path_parts
            nested_passwords = self.passwords[1:] if self.passwords and len(self.passwords) > 1 else None

            try:
                with self.zip_file.open(nested_archive_name) as nested_archive_file:
                    nested_archive_data = nested_archive_file.read()
                    nested_archive_in_memory = io.BytesIO(nested_archive_data)
                    nested_handler = ZipArchiveHandler(nested_archive_in_memory, nested_passwords)
                    return nested_handler.open_file(path_in_nested_archive)
            except (RuntimeError, KeyError, IOError) as e:
                raise IOError(f"Failed to open nested archive {nested_archive_name}: {e}") from e


def register() -> None:
    register_handler(ZipArchiveHandler)


register()
