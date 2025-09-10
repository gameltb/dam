import zipfile
from typing import IO, BinaryIO, List, Optional, Union

from ..base import ArchiveHandler, ArchiveMemberInfo
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

    def list_files(self) -> List[ArchiveMemberInfo]:
        file_list: List[ArchiveMemberInfo] = []
        for f in self.zip_file.infolist():
            if f.is_dir():
                continue

            filename: str = f.filename
            # ZIP spec says that if the 8th bit of the general purpose bit flag is set,
            # the filename is encoded in UTF-8.
            # Otherwise, it's encoded in CP437.
            # https://stackoverflow.com/questions/37723505/namelist-from-zipfile-returns-strings-with-an-invalid-encoding
            ZIP_FILENAME_UTF8_FLAG = 0x800
            if f.flag_bits & ZIP_FILENAME_UTF8_FLAG == 0:
                try:
                    # The filename is decoded with cp437 by default by zipfile module.
                    # We re-encode it to get the original bytes.
                    filename_bytes = filename.encode("cp437")
                    # Then, we try to decode it with common encodings.
                    # UTF-8 is tried first, which can handle most cases if the creating system was modern.
                    # GBK is a common encoding for Chinese systems.
                    try:
                        filename = filename_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        filename = filename_bytes.decode("gbk")
                except Exception:
                    # If any error occurs, we just fall back to the default filename provided by zipfile.
                    pass
            file_list.append(ArchiveMemberInfo(name=filename, size=f.file_size))
        return file_list

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
