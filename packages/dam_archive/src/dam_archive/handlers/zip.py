import zipfile
from typing import IO, BinaryIO, Dict, Iterable, Iterator, List, Optional, Union, cast

from ..base import ArchiveFile, ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError


class ZipArchiveFile(ArchiveFile):
    """
    Represents a file within a zip archive.
    """

    def __init__(
        self, zip_file: zipfile.ZipFile, zip_info: zipfile.ZipInfo, decoded_name: str, password: Optional[str] = None
    ):
        self._zip_file = zip_file
        self._zip_info = zip_info
        self._decoded_name = decoded_name
        self._password = password
        self._fileobj: Optional[BinaryIO] = None
        self._closed = False

    def _get_fileobj(self) -> BinaryIO:
        if self._fileobj is None:
            try:
                pwd = self._password.encode() if self._password else None
                self._fileobj = cast(BinaryIO, self._zip_file.open(self._zip_info, pwd=pwd))
            except RuntimeError as e:
                if "password" in str(e).lower():
                    raise InvalidPasswordError("Invalid password for zip file.") from e
                raise IOError(f"Failed to open file in zip: {e}") from e
        assert self._fileobj is not None
        return self._fileobj

    @property
    def name(self) -> str:
        return self._decoded_name

    @property
    def size(self) -> int:
        return self._zip_info.file_size

    def read(self, n: int = -1) -> bytes:
        return self._get_fileobj().read(n)

    def fileno(self) -> int:
        raise OSError("fileno is not supported")

    def flush(self) -> None:
        if self._fileobj is not None:
            self._fileobj.flush()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def readline(self, limit: int = -1) -> bytes:
        return self._get_fileobj().readline(limit)

    def readlines(self, hint: int = -1) -> list[bytes]:
        return self._get_fileobj().readlines(hint)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._get_fileobj().seek(offset, whence)

    def seekable(self) -> bool:
        return self._get_fileobj().seekable()

    def tell(self) -> int:
        return self._get_fileobj().tell()

    def truncate(self, size: Optional[int] = None) -> int:
        raise OSError("truncate is not supported")

    def writable(self) -> bool:
        return False

    def write(self, s: bytes) -> int:  # type: ignore[override]
        raise OSError("write is not supported")

    def writelines(self, lines: Iterable[bytes]) -> None:  # type: ignore[override]
        raise OSError("writelines is not supported")

    def close(self) -> None:
        if not self._closed:
            if self._fileobj:
                try:
                    self._fileobj.close()
                except Exception:
                    pass
            self._closed = True

    def __enter__(self) -> "ZipArchiveFile":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.close()

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __del__(self) -> None:
        self.close()


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
        if isinstance(self.file, IO):
            try:
                self.file.close()
            except Exception:
                pass

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[ArchiveFile]:
        """Iterate over all files in the archive."""
        for f in self.zip_file.infolist():
            if f.is_dir():
                continue
            decoded_name = self._decode_zip_filename(f)
            yield ZipArchiveFile(self.zip_file, f, decoded_name, self.password)

    def open_file(self, file_name: str) -> ArchiveFile:
        original_name = self.filename_map.get(file_name)
        if not original_name:
            raise IOError(f"File not found in zip: {file_name}")
        for f in self.zip_file.infolist():
            if f.filename == original_name:
                return ZipArchiveFile(self.zip_file, f, file_name, self.password)
        raise IOError(f"File not found in zip: {file_name}")
