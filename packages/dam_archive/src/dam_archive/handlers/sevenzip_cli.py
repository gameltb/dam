from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from ..base import ArchiveHandler, ArchiveMemberInfo, StreamProvider
from ..exceptions import ArchiveError, InvalidPasswordError

logger = logging.getLogger(__name__)


class SevenZipCliArchiveHandler(ArchiveHandler):
    """
    An archive handler that uses the 7z command-line tool for extraction.
    This handler is intended to support archives with filters that py7zr does not,
    such as BCJ2.
    """

    FILE_LIST_RE = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"  # Date and time
        r"([DRHSA.]{5})\s+"  # Attributes
        r"(\d+)\s+"  # Size
        r"(\d*)\s+"  # Compressed size
        r"(.+)"  # Name
    )

    def __init__(
        self,
        stream_provider: StreamProvider,
        password: str | None,
        file_path: str,
        temp_file_path: str | None,
    ):
        super().__init__(stream_provider, password)
        self.file_path = file_path
        self.members: list[ArchiveMemberInfo] = []
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._temp_file_path = temp_file_path

        if self.password:
            self._validate_password()
        self._list_files_and_populate_members()

    @classmethod
    async def create(cls, stream_provider: StreamProvider, password: str | None = None) -> SevenZipCliArchiveHandler:
        temp_file_path: str | None = None
        path: Path | None = stream_provider.get_path()

        if path is None:
            # The 7z CLI tool requires a file path, so we must write the stream to a temporary file.
            fd, temp_path_str = tempfile.mkstemp(suffix=".7z")
            temp_file_path = temp_path_str
            file_path = Path(temp_path_str)

            try:
                async with stream_provider.get_stream() as stream:
                    with os.fdopen(fd, "wb") as temp_file:
                        shutil.copyfileobj(stream, temp_file)
            except Exception:
                # Ensure cleanup if something goes wrong during stream copying
                Path(temp_path_str).unlink()
                raise
        else:
            file_path = path

        return cls(stream_provider, password, str(file_path), temp_file_path)

    def _validate_password(self):
        args = ["t", self.file_path]
        if self.password:
            args.append(f"-p{self.password}")

        try:
            self._run_7z(args)
        except ArchiveError as e:
            if "wrong password" in str(e).lower():
                raise InvalidPasswordError("Invalid password for 7z archive.") from e
            raise

    def _run_7z(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["7z", *args]
        try:
            return subprocess.run(
                command,
                capture_output=True,
                check=check,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as e:
            raise ArchiveError("7z command not found. Please ensure it is installed and in your PATH.") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.lower()
            if "wrong password" in stderr or "data error in encrypted file" in stderr:
                raise InvalidPasswordError("Invalid password for 7z archive.") from e
            if "can not open the file as archive" in stderr:
                raise ArchiveError(f"File is not a valid 7z archive or is corrupted: {self.file_path}") from e
            raise ArchiveError(f"7z command failed with exit code {e.returncode}: {e.stderr}") from e

    def _list_files_and_populate_members(self) -> None:
        args = ["l", "-ba", self.file_path]
        if self.password:
            args.append(f"-p{self.password}")

        result = self._run_7z(args)
        self._parse_file_list(result.stdout)

    def _parse_file_list(self, output: str) -> None:
        """Parses the output of '7z l -ba'."""
        members: list[ArchiveMemberInfo] = []
        for line in output.splitlines():
            match = self.FILE_LIST_RE.match(line)
            if match:
                modified_at_str, attrs, size_str, _, name = match.groups()
                if "D" in attrs:  # Skip directories
                    continue

                try:
                    modified_at = datetime.strptime(modified_at_str, "%Y-%m-%d %H:%M:%S")
                    size = int(size_str) if size_str else 0

                    # Normalize path separators to forward slashes
                    name = name.replace("\\", "/")

                    members.append(ArchiveMemberInfo(name=name, size=size, modified_at=modified_at))
                except (ValueError, TypeError) as e:
                    logger.warning("Could not parse file list line: %s (%s)", line, e)

        self.members = members

    def list_files(self) -> list[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[tuple[ArchiveMemberInfo, BinaryIO]]:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="dam_7z_")

        extract_path = self._temp_dir.name
        args = ["x", f"-o{extract_path}", self.file_path, "-y"]
        if self.password:
            args.append(f"-p{self.password}")

        self._run_7z(args)

        # Verify that the extracted files match the members list
        extracted_files = {
            str(p.relative_to(extract_path)).replace("\\", "/") for p in Path(extract_path).rglob("*") if p.is_file()
        }
        member_files = {member.name for member in self.members}

        if extracted_files != member_files:
            logger.warning(
                "Mismatch between archive members and extracted files.\n"
                "Members not found on disk: %s\n"
                "Files on disk not in members: %s",
                sorted(list(member_files - extracted_files)),
                sorted(list(extracted_files - member_files)),
            )

        for member in self.members:
            file_path = Path(extract_path) / member.name
            if file_path.is_file():
                file_handle = file_path.open("rb")
                yield member, file_handle  # pyright: ignore[reportReturnType]
            else:
                logger.warning("File '%s' not found in extraction directory '%s'.", member.name, extract_path)

    def open_file(self, file_name: str) -> tuple[ArchiveMemberInfo, BinaryIO]:
        member_info = next((m for m in self.members if m.name == file_name), None)
        if not member_info:
            raise OSError(f"File not found in 7z archive: {file_name}")

        args = ["e", "-so", self.file_path, file_name]
        if self.password:
            args.append(f"-p{self.password}")

        try:
            process = subprocess.Popen(
                ["7z", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise ArchiveError("7z command not found. Please ensure it is installed and in your PATH.") from e

        if process.stdout is None:
            raise ArchiveError("Failed to open file stream from 7z.")

        return member_info, process.stdout  # pyright: ignore[reportReturnType]

    async def close(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
        if self._temp_file_path:
            try:
                Path(self._temp_file_path).unlink()
                self._temp_file_path = None
            except OSError as e:
                logger.warning("Could not delete temporary file '%s': %s", self._temp_file_path, e)
