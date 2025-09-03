from typing import Optional

from .base import ArchiveHandler
from .registry import get_handlers


def open_archive(file_path: str, password: Optional[str] = None) -> Optional[ArchiveHandler]:
    """
    Open an archive file with the appropriate handler.

    Args:
        file_path: The path to the archive file.
        password: The password for the archive, if any.

    Returns:
        An instance of an ArchiveHandler, or None if no suitable handler is found.
    """
    for handler_class in get_handlers():
        if handler_class.can_handle(file_path):
            return handler_class(file_path, password)
    return None
