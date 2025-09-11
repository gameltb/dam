from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import IO, BinaryIO, Iterator, List, Optional, Union


@dataclass
class ArchiveMemberInfo:
    """
    Represents information about a member in an archive.
    """

    name: str
    size: int


class ArchiveFile(IO[bytes], ABC):
    """
    Represents a file within an archive.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the file."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the file in bytes."""
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
    def iter_files(self) -> Iterator[ArchiveFile]:
        """
        Iterate over all files in the archive in their natural order.

        This method is designed for efficient, sequential processing of archive
        members. It yields `ArchiveFile` objects, which can be used to access
        member information and open a stream to the file's content.

        For archive formats that support it, this method should be implemented
        to stream data from the archive rather than performing random-access reads,
        which can be inefficient, especially for solid archives.
        """
        pass

    @abstractmethod
    def open_file(self, file_name: str) -> ArchiveFile:
        """Open a specific file from the archive and return a file-like object."""

        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the archive file and releases any resources.
        This should be called when the handler is no longer needed.
        """
        pass
