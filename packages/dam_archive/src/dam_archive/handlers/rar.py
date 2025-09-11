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
        self._fileobj: Optional[IO[bytes]] = None
        self._closed = False

    @property
    def name(self) -> str:
        return self._rar_info.filename  # type: ignore

    @property
    def size(self) -> int:
        return self._rar_info.file_size  # type: ignore

    def read(self, *args: object, **kwargs: object) -> bytes:
        if self._fileobj is None:
            try:
                self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
            except rarfile.WrongPassword as e:  # type: ignore
                raise InvalidPasswordError(
                    f"Invalid password for file '{self._rar_info.filename}' in rar archive."
                ) from e
            except KeyError as e:
                raise IOError(f"File not found in rar: {self._rar_info.filename}") from e
        return self._fileobj.read(*args, **kwargs)

    def fileno(self) -> int:
        raise OSError("fileno is not supported")

    def flush(self) -> None:
        if self._fileobj is not None:
            self._fileobj.flush()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def readline(self, *args: object, **kwargs: object) -> bytes:
        if self._fileobj is None:
            self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
        return self._fileobj.readline(*args, **kwargs)

    def readlines(self, *args: object, **kwargs: object) -> list[bytes]:
        if self._fileobj is None:
            self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
        return self._fileobj.readlines(*args, **kwargs)

    def seek(self, offset: int, whence: int = 0) -> int:
        if self._fileobj is None:
            self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
        return self._fileobj.seek(offset, whence)

    def seekable(self) -> bool:
        if self._fileobj is None:
            self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
        return self._fileobj.seekable()

    def tell(self) -> int:
        if self._fileobj is None:
            self._fileobj = self._rar_file.open(self._rar_info)  # type: ignore
        return self._fileobj.tell()

    def truncate(self, size: Optional[int] = None) -> int:
        raise OSError("truncate is not supported")

    def writable(self) -> bool:
        return False

    def write(self, *args: object, **kwargs: object) -> int:
        raise OSError("write is not supported")

    def writelines(self, lines) -> None:
        raise OSError("writelines is not supported")

    def close(self):
        if not self._closed:
            if self._fileobj:
                try:
                    self._fileobj.close()
                except Exception:
                    pass
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.close()

    def __del__(self):
        self.close()


class RarArchiveHandler(ArchiveHandler):
    def close(self) -> None:
        try:
            self.rar_file.close()
        except Exception:
            pass
        if hasattr(self.file, "close"):
            try:
                self.file.close()
            except Exception:
                pass

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

    def open_file(self, file_name: str) -> ArchiveFile:
        for f in self.rar_file.infolist():  # type: ignore
            if f.filename == file_name:
                return RarArchiveFile(self.rar_file, f)
        raise IOError(f"File not found in rar: {file_name}")


def register() -> None:
    register_handler(RarArchiveHandler)
