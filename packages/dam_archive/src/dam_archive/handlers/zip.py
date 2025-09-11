import zipfile
from typing import IO, BinaryIO, Dict, Iterator, List, Optional, Union

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError
from ..registry import register_handler


class ZipArchiveFile(ArchiveFile):
    """
    Represents a file within a zip archive.
    """

    def __init__(
        self,
        zip_file: zipfile.ZipFile,
        zip_info: zipfile.ZipInfo,
        decoded_name: str,
        password: Optional[str] = None,
    ):
        self._zip_file = zip_file
        self._zip_info = zip_info
        self._decoded_name = decoded_name
        self._password = password

    @property
    def name(self) -> str:
        return self._decoded_name

    @property
    def size(self) -> int:
        return self._zip_info.file_size

    def open(self) -> IO[bytes]:
        try:
            pwd = self._password.encode() if self._password else None
            return self._zip_file.open(self._zip_info, pwd=pwd)
        except RuntimeError as e:
            if "password" in str(e).lower():
                raise InvalidPasswordError("Invalid password for zip file.") from e
            raise IOError(f"Failed to open file in zip: {e}") from e


class ZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for zip files.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        self.members: List[ArchiveMemberInfo] = []
        self.filename_map: Dict[str, str] = {}
        try:
            self.zip_file = zipfile.ZipFile(self.file, "r")

            for f in self.zip_file.infolist():
                if f.is_dir():
                    continue

                original_name = f.filename
                decoded_name: str = self._decode_zip_filename(f)
                self.filename_map[decoded_name] = original_name
                self.members.append(ArchiveMemberInfo(name=decoded_name, size=f.file_size))

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

    def _decode_zip_filename(self, info: zipfile.ZipInfo) -> str:
        filename = info.filename
        # ZIP spec says that if the 8th bit of the general purpose bit flag is set,
        # the filename is encoded in UTF-8.
        # Otherwise, it's encoded in CP437.
        # https://stackoverflow.com/questions/37723505/namelist-from-zipfile-returns-strings-with-an-invalid-encoding
        ZIP_FILENAME_UTF8_FLAG = 0x800
        if info.flag_bits & ZIP_FILENAME_UTF8_FLAG == 0:
            try:
                # The filename is decoded with cp437 by default by zipfile module.
                # We re-encode it to get the original bytes.
                filename_bytes = filename.encode("cp437")
                # Then, we try to decode it with common encodings.
                # UTF-8 is tried first, which can handle most cases if the creating system was modern.
                # GBK is a common encoding for Chinese systems.
                try:
                    return filename_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return filename_bytes.decode("gbk")
            except Exception:
                # If any error occurs, we just fall back to the default filename provided by zipfile.
                return filename
        return filename

    @staticmethod
    def can_handle(file_path: str) -> bool:
        return file_path.lower().endswith(".zip")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[ArchiveFile]:
        """Iterate over all files in the archive."""
        for f in self.zip_file.infolist():
            if f.is_dir():
                continue
            decoded_name = self._decode_zip_filename(f)
            yield ZipArchiveFile(self.zip_file, f, decoded_name, self.password)

    def open_file(self, file_name: str) -> IO[bytes]:
        original_name = self.filename_map.get(file_name)
        if not original_name:
            raise IOError(f"File not found in zip: {file_name}")

        try:
            pwd = self.password.encode() if self.password else None
            return self.zip_file.open(original_name, pwd=pwd)
        except RuntimeError as e:
            if "password" in str(e).lower():
                raise InvalidPasswordError("Invalid password for zip file.") from e
            raise IOError(f"Failed to open file in zip: {e}") from e
        except KeyError as e:
            # This should not happen if the filename_map is correct
            raise IOError(f"File not found in zip: {file_name}") from e


def register() -> None:
    register_handler(ZipArchiveHandler)


register()
