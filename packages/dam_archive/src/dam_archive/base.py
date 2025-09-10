from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import IO, BinaryIO, List, Optional, Union


@dataclass
class ArchiveMemberInfo:
    """
    Represents information about a member in an archive.
    """

    name: str
    size: int


class ArchiveFile(ABC):
    """
    Represents a file within an archive.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the file."""
        pass

    @abstractmethod
    def open(self) -> IO[bytes]:
        """Open the file and return a file-like object."""
        pass


class ArchiveHandler(ABC):
    """
    Abstract base class for archive handlers.
    """

    @staticmethod
    @abstractmethod
    def can_handle(file_path: str) -> bool:
        """Check if this handler can open the given file."""
        pass

    @abstractmethod
    def __init__(self, file: Union[str, BinaryIO], password: Optional[str] = None):
        """
        Initializes the archive handler.

        Args:
            file: The path to the archive file or a file-like object.
            password: The password for the archive, if any.
        """
        pass

    @abstractmethod
    def list_files(self) -> List[ArchiveMemberInfo]:
        """List all file names and sizes in the archive."""
        pass

    @abstractmethod
    def open_file(self, file_name: str) -> IO[bytes]:
        """Open a specific file from the archive and return a file-like object."""
        pass
