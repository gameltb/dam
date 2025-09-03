from abc import ABC, abstractmethod
from typing import IO, List, Optional


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
    def __init__(self, file_path: str, password: Optional[str] = None):
        """
        Initializes the archive handler.

        Args:
            file_path: The path to the archive file.
            password: The password for the archive, if any.
        """
        pass

    @abstractmethod
    def list_files(self) -> List[str]:
        """List all file names in the archive."""
        pass

    @abstractmethod
    def open_file(self, file_name: str) -> IO[bytes]:
        """Open a specific file from the archive and return a file-like object."""
        pass
