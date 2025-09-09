from typing import IO, Any, List, Optional

import rarfile

from .base import ArchiveHandler
from .registry import register_handler


class RarArchiveHandler(ArchiveHandler):
    """
    An archive handler for rar files.
    """

    import shutil
    import tempfile

    def __init__(self, file: Any, passwords: Optional[List[str]] = None):
        self.file = file
        self.passwords = passwords

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".rar")

    from contextlib import contextmanager

    @contextmanager
    def _open_archive_with_temp_file(self, pwd: Optional[str] = None):
        tmp_path = None
        if isinstance(self.file, str):
            rar_file = rarfile.RarFile(self.file, "r")
        else:
            self.file.seek(0)
            with self.tempfile.NamedTemporaryFile(delete=False) as tmp:
                self.shutil.copyfileobj(self.file, tmp)
                tmp_path = tmp.name
            rar_file = rarfile.RarFile(tmp_path, "r")

        if pwd:
            rar_file.setpassword(pwd)  # type: ignore

        try:
            yield rar_file
        finally:
            rar_file.close()
            if tmp_path:
                import os

                os.unlink(tmp_path)

    def _try_open(self):
        if self.passwords:
            for pwd in self.passwords:
                try:
                    return self._open_archive_with_temp_file(pwd)
                except rarfile.BadRarFile:
                    continue
        return self._open_archive_with_temp_file()

    def list_files(self) -> List[str]:
        with self._try_open() as archive:
            return [f.filename for f in archive.infolist() if not f.isdir()]  # type: ignore

    def open_file(self, file_name: str) -> IO[bytes]:
        with self._try_open() as archive:
            return archive.open(file_name)  # type: ignore


def register() -> None:
    register_handler(RarArchiveHandler)
