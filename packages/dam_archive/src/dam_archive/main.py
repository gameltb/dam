from typing import Optional, List

from .base import ArchiveHandler
from .registry import get_handlers


def open_archive(file_path: str, passwords: Optional[List[str]] = None) -> Optional[ArchiveHandler]:
    """
    Open an archive file with the appropriate handler.

    Args:
        file_path: The path to the archive file.
        passwords: The passwords for the archive, if any.

    Returns:
        An instance of an ArchiveHandler, or None if no suitable handler isfound.
    """
    for handler_class in get_handlers():
        if handler_class.can_handle(file_path):
            return handler_class(file_path, passwords)
    return None
