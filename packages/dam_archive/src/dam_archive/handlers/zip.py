import zipfile
from datetime import datetime
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple, cast

from ..base import ArchiveHandler, ArchiveMemberInfo, StreamProvider
from ..exceptions import InvalidPasswordError


class ZipArchiveHandler(ArchiveHandler):
    """
    An archive handler for zip files.
    """

    def __init__(self, stream_provider: StreamProvider, password: Optional[str] = None):
        super().__init__(stream_provider, password)
        self.members: List[ArchiveMemberInfo] = []
        self.filename_map: Dict[str, str] = {}
        self._stream: Optional[BinaryIO] = None
        try:
            self._stream = self._stream_provider()
            self.zip_file = zipfile.ZipFile(self._stream, "r")

            for f in self.zip_file.infolist():
                if f.is_dir():
                    continue

                original_name = f.filename
                decoded_name: str = self._decode_zip_filename(f)
                self.filename_map[decoded_name] = original_name
                modified_at = datetime(*f.date_time)
                self.members.append(ArchiveMemberInfo(name=decoded_name, size=f.file_size, modified_at=modified_at))

            if self.password:
                if not self.zip_file.infolist():
                    return  # No files to check
                # Try to open the first file to check password
                with self.zip_file.open(self.zip_file.infolist()[0], pwd=self.password.encode()) as f:
                    f.read(1)
        except RuntimeError:
            raise InvalidPasswordError("Invalid password for zip file.")
        except zipfile.BadZipFile as e:
            raise InvalidPasswordError(f"Invalid password for zip file: {e}") from e

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

    def close(self) -> None:
        try:
            self.zip_file.close()
        except Exception:
            pass
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[Tuple[ArchiveMemberInfo, BinaryIO]]:
        """Iterate over all files in the archive."""
        for f in self.zip_file.infolist():
            if f.is_dir():
                continue
            decoded_name = self._decode_zip_filename(f)
            member_info = ArchiveMemberInfo(name=decoded_name, size=f.file_size, modified_at=datetime(*f.date_time))
            try:
                pwd = self.password.encode() if self.password else None
                yield member_info, cast(BinaryIO, self.zip_file.open(f, pwd=pwd))
            except RuntimeError as e:
                if "password" in str(e).lower():
                    raise InvalidPasswordError("Invalid password for zip file.") from e
                raise IOError(f"Failed to open file in zip: {e}") from e

    def open_file(self, file_name: str) -> Tuple[ArchiveMemberInfo, BinaryIO]:
        original_name = self.filename_map.get(file_name)
        if not original_name:
            raise IOError(f"File not found in zip: {file_name}")
        for f in self.zip_file.infolist():
            if f.filename == original_name:
                member_info = ArchiveMemberInfo(name=file_name, size=f.file_size, modified_at=datetime(*f.date_time))
                try:
                    pwd = self.password.encode() if self.password else None
                    return member_info, cast(BinaryIO, self.zip_file.open(f, pwd=pwd))
                except RuntimeError as e:
                    if "password" in str(e).lower():
                        raise InvalidPasswordError("Invalid password for zip file.") from e
                    raise IOError(f"Failed to open file in zip: {e}") from e
        raise IOError(f"File not found in zip: {file_name}")

    def _decode_comment(self, comment: bytes) -> str:
        try:
            return comment.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return comment.decode("gbk")
            except UnicodeDecodeError:
                return comment.decode("cp437")

    @property
    def comment(self) -> Optional[str]:
        if not self.zip_file.comment:
            return None
        return self._decode_comment(self.zip_file.comment)
