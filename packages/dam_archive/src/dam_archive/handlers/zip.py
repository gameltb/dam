import zipfile
from typing import IO, BinaryIO, List, Optional, Union

from ..base import ArchiveHandler
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


class ZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for zip files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        try:
            self.zip_file = zipfile.ZipFile(self.file, "r")
            if self.password:
                if not self.zip_file.infolist():
                    return  # No files to check
                # Try to open the first file to check password
                with self.zip_file.open(self.zip_file.infolist()[0], pwd=self.password.encode()) as f:
                    f.read(1)
        except RuntimeError:
            # The zipfile module is not consistent in the exception it raises for
            # incorrect passwords. It's supposed to be a RuntimeError with "Bad password"
            # in the message, but in some cases it raises other RuntimeErrors.
            # For our purpose, we'll assume any RuntimeError during this check is
            # due to an incorrect password.
            raise InvalidPasswordError("Invalid password for zip file.")
        except zipfile.BadZipFile as e:
            raise IOError(f"Failed to open zip file: {e}") from e

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".zip")

    def list_files(self) -> List[str]:
        return [f.filename for f in self.zip_file.infolist() if not f.is_dir()]

    def open_file(self, file_name: str) -> IO[bytes]:
        try:
            pwd = self.password.encode() if self.password else None
            return self.zip_file.open(file_name, pwd=pwd)
        except RuntimeError as e:
            if "password" in str(e).lower():
                raise InvalidPasswordError("Invalid password for zip file.") from e
            raise IOError(f"Failed to open file in zip: {e}") from e
        except KeyError as e:
            raise IOError(f"File not found in zip: {file_name}") from e


def register() -> None:
    register_handler(ZipArchiveHandler)


register()
