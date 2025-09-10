from typing import IO, BinaryIO, List, Optional, Union

import rarfile

from ..base import ArchiveHandler
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


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

    def list_files(self) -> List[str]:
        return [f.filename for f in self.rar_file.infolist() if not f.isdir()]  # type: ignore

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
