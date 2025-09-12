from typing import IO, BinaryIO, Iterable, Iterator, List, Optional, Union

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
        self._fileobj: Optional[BinaryIO] = None
        self._closed = False

    def _get_fileobj(self) -> BinaryIO:
        if self._fileobj is None:
            try:
                self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
            except rarfile.WrongPassword as e:  # type: ignore
                raise InvalidPasswordError(
                    f"Invalid password for file '{self._rar_info.filename}' in rar archive."
                ) from e
            except KeyError as e:
                raise IOError(f"File not found in rar: {self._rar_info.filename}") from e
        assert self._fileobj is not None
        return self._fileobj

    @property
    def name(self) -> str:
        return self._rar_info.filename  # type: ignore

    @property
    def size(self) -> int:
        return self._rar_info.file_size  # type: ignore

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

    def __enter__(self) -> "RarArchiveFile":
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
            if self.rar_file.infolist():
                self.rar_file.infolist()[0].filename  # type: ignore

        except rarfile.WrongPassword as e:  # type: ignore
            raise InvalidPasswordError("Invalid password for rar file.") from e
        except rarfile.BadRarFile as e:
            raise IOError(f"Failed to open rar file: {e}") from e

    def close(self) -> None:
        try:
            self.rar_file.close()
        except Exception:
            pass
        if isinstance(self.file, IO):
            try:
                self.file.close()
            except Exception:
                pass

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

    def open_file(self, file_name: str) -> ArchiveFile:
        for f in self.rar_file.infolist():  # type: ignore
            if f.filename == file_name:  # type: ignore
                return RarArchiveFile(self.rar_file, f)  # type: ignore
        raise IOError(f"File not found in rar: {file_name}")


def register() -> None:
    register_handler(RarArchiveHandler)


register()
