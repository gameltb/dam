from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import (
    Awaitable,
    BinaryIO,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)

import rarfile

from ..base import ArchiveHandler, ArchiveMemberInfo, StreamProvider
from ..exceptions import InvalidPasswordError


class RarArchiveHandler(ArchiveHandler):
    """
    An archive handler for rar files.
    """

    def __init__(self, stream_provider: StreamProvider, password: Optional[str] = None):
        super().__init__(stream_provider, password)
        self._stream: Optional[BinaryIO] = None
        self.rar_file: Optional[rarfile.RarFile] = None
        self._stream_cm_exit: Optional[
            Callable[
                [
                    Optional[type[BaseException]],
                    Optional[BaseException],
                    Optional[TracebackType],
                ],
                Awaitable[Optional[bool]],
            ]
        ] = None

    @classmethod
    async def create(cls, stream_provider: StreamProvider, password: Optional[str] = None) -> RarArchiveHandler:
        handler = cls(stream_provider, password)
        stream_cm = stream_provider.get_stream()
        handler._stream = await stream_cm.__aenter__()
        handler._stream_cm_exit = stream_cm.__aexit__

        try:
            handler.rar_file = rarfile.RarFile(handler._stream, "r")  # type: ignore

            if password:
                handler.rar_file.setpassword(password)  # type: ignore
            # The check below is to trigger a password error for protected archives.
            if len(handler.rar_file.infolist()) > 0:  # type: ignore
                handler.rar_file.infolist()[0].filename  # type: ignore

        except Exception as e:
            if handler._stream_cm_exit:
                await handler._stream_cm_exit(type(e), e, e.__traceback__)

            if isinstance(e, rarfile.RarWrongPassword):  # type: ignore
                raise InvalidPasswordError("Invalid password for rar file.") from e
            elif isinstance(e, rarfile.BadRarFile):  # type: ignore
                raise IOError(f"Failed to open rar file: {e}") from e
            else:
                raise

        return handler

    async def close(self) -> None:
        if self.rar_file:
            try:
                self.rar_file.close()
            except Exception:
                pass

        if self._stream_cm_exit:
            await self._stream_cm_exit(None, None, None)

        self._stream = None
        self._stream_cm_exit = None

    def list_files(self) -> List[ArchiveMemberInfo]:
        files: List[ArchiveMemberInfo] = []
        if not self.rar_file:
            return files

        for f in self.rar_file.infolist():  # type: ignore
            if not f.isdir():  # type: ignore
                mtime = cast(Optional[datetime], f.mtime)  # type: ignore
                if mtime is None:
                    mtime = rarfile.to_datetime(f.date_time)  # type: ignore
                files.append(
                    ArchiveMemberInfo(
                        name=f.filename,  # type: ignore
                        size=f.file_size,  # type: ignore
                        modified_at=mtime,
                    )
                )
        return files

    def iter_files(self) -> Iterator[Tuple[ArchiveMemberInfo, BinaryIO]]:
        """Iterate over all files in the archive."""
        if not self.rar_file:
            return

        for f in self.rar_file.infolist():  # type: ignore
            if f.isdir():  # type: ignore
                continue
            mtime = cast(Optional[datetime], f.mtime)  # type: ignore
            if mtime is None:
                mtime = rarfile.to_datetime(f.date_time)  # type: ignore
            member_info = ArchiveMemberInfo(
                name=f.filename,  # type: ignore
                size=f.file_size,  # type: ignore
                modified_at=mtime,
            )
            try:
                yield member_info, self.rar_file.open(f)  # type: ignore
            except rarfile.WrongPassword as e:  # type: ignore
                raise InvalidPasswordError(f"Invalid password for file '{f.filename}' in rar archive.") from e  # type: ignore
            except KeyError as e:
                raise IOError(f"File not found in rar: {f.filename}") from e  # type: ignore

    def open_file(self, file_name: str) -> Tuple[ArchiveMemberInfo, BinaryIO]:
        if not self.rar_file:
            raise IOError(f"File not found in rar: {file_name}")

        for f in self.rar_file.infolist():  # type: ignore
            if f.filename == file_name:  # type: ignore
                mtime = cast(Optional[datetime], f.mtime)  # type: ignore
                if mtime is None:
                    mtime = rarfile.to_datetime(f.date_time)  # type: ignore
                member_info = ArchiveMemberInfo(
                    name=f.filename,  # type: ignore
                    size=f.file_size,  # type: ignore
                    modified_at=mtime,
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
        if not self.rar_file:
            return None
        comment = self.rar_file.comment  # type: ignore
        if comment and "\x00" in comment:
            try:
                return comment.encode("raw_unicode_escape").decode("utf-16-le")  # type: ignore
            except Exception:
                return comment  # type: ignore
        return comment  # type: ignore
