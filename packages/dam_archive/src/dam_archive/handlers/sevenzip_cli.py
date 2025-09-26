import logging
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import IO, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

from ..base import ArchiveHandler, ArchiveMemberInfo
from ..exceptions import ArchiveError, InvalidPasswordError

logger = logging.getLogger(__name__)


class SevenZipCliArchiveHandler(ArchiveHandler):
    """
    An archive handler that uses the 7z command-line tool for extraction.
    This handler is intended to support archives with filters that py7zr does not,
    such as BCJ2.
    """

    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        self.file = file
        self.password = password
        self.members: List[ArchiveMemberInfo] = []
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._temp_file_path: Optional[str] = None

        if isinstance(self.file, str):
            self.file_path = self.file
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".7z") as temp_file:
                self.file_path = temp_file.name
                self._temp_file_path = self.file_path
                if isinstance(self.file, IO):
                    while True:
                        chunk = self.file.read(8192)
                        if not chunk:
                            break
                        temp_file.write(chunk)
                else:
                    raise ArchiveError("Input file object does not have a read method.")

        if self.password:
            self._validate_password()
        self._list_files_and_populate_members()

    def _validate_password(self):
        args = ["t", self.file_path]
        if self.password:
            args.append(f"-p{self.password}")

        try:
            self._run_7z(args)
        except ArchiveError as e:
            if "wrong password" in str(e).lower():
                raise InvalidPasswordError("Invalid password for 7z archive.")
            raise

    def _run_7z(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["7z"] + args
        try:
            return subprocess.run(
                command,
                capture_output=True,
                check=check,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            raise ArchiveError("7z command not found. Please ensure it is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.lower()
            if "wrong password" in stderr or "data error in encrypted file" in stderr:
                raise InvalidPasswordError("Invalid password for 7z archive.")
            if "can not open the file as archive" in stderr:
                raise ArchiveError(f"File is not a valid 7z archive or is corrupted: {self.file_path}")
            raise ArchiveError(f"7z command failed with exit code {e.returncode}: {e.stderr}")

    def _list_files_and_populate_members(self) -> None:
        args = ["l", "-slt", self.file_path]
        if self.password:
            args.append(f"-p{self.password}")

        result = self._run_7z(args)
        self._parse_list_output(result.stdout)

    def _parse_list_output(self, output: str) -> None:
        self.members = []
        file_sections = re.split(r"\n----------\n", output)

        for section in file_sections:
            if not section.strip() or "Path =" not in section:
                continue

            info: Dict[str, str] = {}
            lines = section.strip().split("\n")
            for line in lines:
                parts = line.split(" = ", 1)
                if len(parts) == 2:
                    info[parts[0].strip()] = parts[1].strip()

            if "Attributes" in info and info["Attributes"].startswith("D"):
                continue

            if "Path" in info and "Size" in info:
                try:
                    name = info["Path"]
                    size = int(info["Size"])
                    modified_str = info.get("Modified")
                    modified_at: Optional[datetime] = None
                    if modified_str:
                        try:
                            if "." in modified_str:
                                main_part, frac_part = modified_str.split(".", 1)
                                modified_str = f"{main_part}.{frac_part[:6]}"
                                modified_at = datetime.strptime(modified_str, "%Y-%m-%d %H:%M:%S.%f")
                            else:
                                modified_at = datetime.strptime(modified_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            logger.warning(f"Could not parse modified time '{modified_str}' for file '{name}'")

                    self.members.append(
                        ArchiveMemberInfo(
                            name=name,
                            size=size,
                            modified_at=modified_at,
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping section due to parsing error: {e}\nSection: {section}")

    def list_files(self) -> List[ArchiveMemberInfo]:
        return self.members

    def iter_files(self) -> Iterator[Tuple[ArchiveMemberInfo, BinaryIO]]:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="dam_7z_")

        extract_path = self._temp_dir.name
        args = ["x", f"-o{extract_path}", self.file_path, "-y"]
        if self.password:
            args.append(f"-p{self.password}")

        self._run_7z(args)

        for member in self.members:
            file_path = Path(extract_path) / member.name
            if file_path.is_file():
                file_handle = open(file_path, "rb")
                yield member, file_handle  # pyright: ignore[reportReturnType]
            else:
                logger.warning(f"File '{member.name}' not found in extraction directory '{extract_path}'.")

    def open_file(self, file_name: str) -> Tuple[ArchiveMemberInfo, BinaryIO]:
        member_info = next((m for m in self.members if m.name == file_name), None)
        if not member_info:
            raise IOError(f"File not found in 7z archive: {file_name}")

        args = ["e", "-so", self.file_path, file_name]
        if self.password:
            args.append(f"-p{self.password}")

        try:
            process = subprocess.Popen(
                ["7z"] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise ArchiveError("7z command not found. Please ensure it is installed and in your PATH.")

        if process.stdout is None:
            raise ArchiveError("Failed to open file stream from 7z.")

        return member_info, process.stdout  # pyright: ignore[reportReturnType]

    def close(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
        if self._temp_file_path:
            try:
                Path(self._temp_file_path).unlink()
                self._temp_file_path = None
            except OSError as e:
                logger.warning(f"Could not delete temporary file '{self._temp_file_path}': {e}")

        if isinstance(self.file, IO):
            try:
                self.file.close()
            except Exception:
                pass
