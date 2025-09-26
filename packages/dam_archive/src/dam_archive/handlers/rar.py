from typing import IO, BinaryIO, Iterator, List, Optional, Tuple, Union

import rarfile

from ..base import ArchiveHandler, ArchiveMemberInfo
from ..exceptions import InvalidPasswordError


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

        except rarfile.RarWrongPassword as e:  # type: ignore
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

    def list_files(self) -> List[ArchiveMemberInfo]:
        files: List[ArchiveMemberInfo] = []
        for f in self.rar_file.infolist():  # type: ignore
            if not f.isdir():  # type: ignore
                mtime = f.mtime  # type: ignore
                if mtime is None:
                    mtime = rarfile.to_datetime(f.date_time)  # type: ignore
                files.append(
                    ArchiveMemberInfo(
                        name=f.filename,  # type: ignore
                        size=f.file_size,  # type: ignore
                        modified_at=mtime,  # type: ignore
                    )
                )
        return files

    def iter_files(self) -> Iterator[Tuple[ArchiveMemberInfo, BinaryIO]]:
        """Iterate over all files in the archive."""
        for f in self.rar_file.infolist():  # type: ignore
            if f.isdir():  # type: ignore
                continue
            mtime = f.mtime  # type: ignore
            if mtime is None:
                mtime = rarfile.to_datetime(f.date_time)  # type: ignore
            member_info = ArchiveMemberInfo(
                name=f.filename,  # type: ignore
                size=f.file_size,  # type: ignore
                modified_at=mtime,  # type: ignore
            )
            try:
                yield member_info, self.rar_file.open(f)  # type: ignore
            except rarfile.WrongPassword as e:  # type: ignore
                raise InvalidPasswordError(f"Invalid password for file '{f.filename}' in rar archive.") from e  # type: ignore
            except KeyError as e:
                raise IOError(f"File not found in rar: {f.filename}") from e  # type: ignore

    def open_file(self, file_name: str) -> Tuple[ArchiveMemberInfo, BinaryIO]:
        for f in self.rar_file.infolist():  # type: ignore
            if f.filename == file_name:  # type: ignore
                mtime = f.mtime  # type: ignore
                if mtime is None:
                    mtime = rarfile.to_datetime(f.date_time)  # type: ignore
                member_info = ArchiveMemberInfo(
                    name=f.filename,  # type: ignore
                    size=f.file_size,  # type: ignore
                    modified_at=mtime,  # type: ignore
                )
                try:
                    return member_info, self.rar_file.open(f)  # type: ignore
                except rarfile.WrongPassword as e:  # type: ignore
                    raise InvalidPasswordError(f"Invalid password for file '{f.filename}' in rar archive.") from e  # type: ignore
                except KeyError as e:
                    raise IOError(f"File not found in rar: {f.filename}") from e  # type: ignore
        raise IOError(f"File not found in rar: {file_name}")

    @property
    def comment(self) -> Optional[str]:
        comment = self.rar_file.comment  # type: ignore
        if comment and "\x00" in comment:
            try:
                return comment.encode("raw_unicode_escape").decode("utf-16-le")  # type: ignore
            except Exception:
                return comment  # type: ignore
        return comment  # type: ignore
