"""Provides a zip archive handler."""

from __future__ import annotations

import contextlib
import zipfile
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime
from types import TracebackType
from typing import (
    BinaryIO,
    cast,
)

from ..base import ArchiveHandler, ArchiveMemberInfo, StreamProvider
from ..exceptions import InvalidPasswordError


class ZipArchiveHandler(ArchiveHandler):
    """An archive handler for zip files."""

    def __init__(self, stream_provider: StreamProvider, password: str | None = None):
        """Initialize the zip archive handler."""
        super().__init__(stream_provider, password)
        self.members: list[ArchiveMemberInfo] = []
        self.filename_map: dict[str, str] = {}
        self._stream: BinaryIO | None = None
        self.zip_file: zipfile.ZipFile | None = None
        self._stream_cm_exit: (
            Callable[[type[BaseException] | None, BaseException | None, TracebackType | None], Awaitable[bool | None]]
            | None
        ) = None

    @classmethod
    async def create(cls, stream_provider: StreamProvider, password: str | None = None) -> ZipArchiveHandler:
        """Asynchronously create and initialize a zip archive handler."""
        handler = cls(stream_provider, password)
        stream_cm = stream_provider.get_stream()
        handler._stream = await stream_cm.__aenter__()
        handler._stream_cm_exit = stream_cm.__aexit__

        try:
            handler.zip_file = zipfile.ZipFile(handler._stream, "r")  # type: ignore

            for f in handler.zip_file.infolist():
                if f.is_dir():
                    continue

                original_name = f.filename
                decoded_name: str = handler._decode_zip_filename(f)
                handler.filename_map[decoded_name] = original_name
                modified_at = datetime(*f.date_time)
                handler.members.append(ArchiveMemberInfo(name=decoded_name, size=f.file_size, modified_at=modified_at))

            if password:
                if len(handler.zip_file.infolist()) == 0:
                    return handler
                with handler.zip_file.open(handler.zip_file.infolist()[0], pwd=password.encode()) as f:
                    f.read(1)
        except Exception as e:
            await handler._stream_cm_exit(type(e), e, e.__traceback__)

            if isinstance(e, RuntimeError):
                raise InvalidPasswordError("Invalid password for zip file.") from e
            if isinstance(e, zipfile.BadZipFile):
                raise InvalidPasswordError(f"Invalid password for zip file: {e}") from e
            raise

        return handler

    def _decode_zip_filename(self, info: zipfile.ZipInfo) -> str:
        filename = info.filename
        # ZIP spec says that if the 8th bit of the general purpose bit flag is set,
        # the filename is encoded in UTF-8.
        # Otherwise, it's encoded in CP437.
        # https://stackoverflow.com/questions/37723505/namelist-from-zipfile-returns-strings-with-an-invalid-encoding
        zip_filename_utf8_flag = 0x800
        if info.flag_bits & zip_filename_utf8_flag == 0:
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

    async def close(self) -> None:
        """Close the zip archive."""
        if self.zip_file:
            with contextlib.suppress(Exception):
                self.zip_file.close()

        if self._stream_cm_exit:
            await self._stream_cm_exit(None, None, None)

        self._stream = None
        self._stream_cm_exit = None

    def list_files(self) -> list[ArchiveMemberInfo]:
        """List all files in the archive."""
        return self.members

    def iter_files(self) -> Iterator[tuple[ArchiveMemberInfo, BinaryIO]]:
        """Iterate over all files in the archive."""
        if not self.zip_file:
            return

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
                raise OSError(f"Failed to open file in zip: {e}") from e

    def open_file(self, file_name: str) -> tuple[ArchiveMemberInfo, BinaryIO]:
        """Open a specific file from the archive and return a file-like object."""
        if not self.zip_file:
            raise OSError(f"File not found in zip: {file_name}")

        original_name = self.filename_map.get(file_name)
        if not original_name:
            raise OSError(f"File not found in zip: {file_name}")
        for f in self.zip_file.infolist():
            if f.filename == original_name:
                member_info = ArchiveMemberInfo(name=file_name, size=f.file_size, modified_at=datetime(*f.date_time))
                try:
                    pwd = self.password.encode() if self.password else None
                    return member_info, cast(BinaryIO, self.zip_file.open(f, pwd=pwd))
                except RuntimeError as e:
                    if "password" in str(e).lower():
                        raise InvalidPasswordError("Invalid password for zip file.") from e
                    raise OSError(f"Failed to open file in zip: {e}") from e
        raise OSError(f"File not found in zip: {file_name}")

    def _decode_comment(self, comment: bytes) -> str:
        try:
            return comment.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return comment.decode("gbk")
            except UnicodeDecodeError:
                return comment.decode("cp437")

    @property
    def comment(self) -> str | None:
        """Return the comment of the zip archive."""
        if not self.zip_file or not self.zip_file.comment:
            return None
        return self._decode_comment(self.zip_file.comment)
