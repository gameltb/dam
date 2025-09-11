from typing import IO, BinaryIO, Iterator, List, Optional, Union

import rarfile

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


class RarArchiveFile(ArchiveFile):
    """
    Represents a file within a rar archive.
    """

    def __init__(self, rar_file: rarfile.RarFile, rar_info: rarfile.RarInfo):
        self._rar_file = rar_file
        self._rar_info = rar_info

    @property
    def name(self) -> str:
        return self._rar_info.filename  # type: ignore

    @property
    def size(self) -> int:
        return self._rar_info.file_size  # type: ignore

    def open(self) -> IO[bytes]:
        try:
            return self._rar_file.open(self._rar_info)  # type: ignore
        except rarfile.WrongPassword as e:  # type: ignore
            raise InvalidPasswordError(f"Invalid password for file '{self._rar_info.filename}' in rar archive.") from e
        except KeyError as e:
            raise IOError(f"File not found in rar: {self._rar_info.filename}") from e


class RarArchiveHandler(ArchiveHandler):
    """
    An archive handler for rar files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password

        try:
            if isinstance(self.file, str):
                self.rar_file = rarfile.RarFile(self.file, "r")
            else:
                if hasattr(self.file, "seek"):
                    self.file.seek(0)
                self.rar_file = rarfile.RarFile(self.file, "r")

            if password:
                self.rar_file.setpassword(password)  # type: ignore
            # The password is not checked until a file is accessed.
            # We will check it by trying to read the first file's info.
            self.rar_file.infolist()[0].filename  # type: ignore

        except rarfile.WrongPassword as e:  # type: ignore
            raise InvalidPasswordError("Invalid password for rar file.") from e
        except rarfile.BadRarFile as e:
            raise IOError(f"Failed to open rar file: {e}") from e

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".rar")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return [ArchiveMemberInfo(name=f.filename, size=f.file_size) for f in self.rar_file.infolist() if not f.isdir()]  # type: ignore

    def iter_files(self) -> Iterator[ArchiveFile]:
        """Iterate over all files in the archive."""
        for f in self.rar_file.infolist():  # type: ignore
            if f.isdir():  # type: ignore
                continue
            yield RarArchiveFile(self.rar_file, f)  # type: ignore

    def open_file(self, file_name: str) -> IO[bytes]:
        try:
            return self.rar_file.open(file_name)  # type: ignore
        except rarfile.WrongPassword as e:  # type: ignore
            # This can happen if the password was correct for the header but not for the file
            raise InvalidPasswordError(f"Invalid password for file '{file_name}' in rar archive.") from e
        except KeyError as e:
            raise IOError(f"File not found in rar: {file_name}") from e


def register() -> None:
    register_handler(RarArchiveHandler)
