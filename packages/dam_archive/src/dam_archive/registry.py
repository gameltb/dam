from typing import List, Type

from .base import ArchiveHandler

_handlers: List[Type[ArchiveHandler]] = []


def register_handler(handler: Type[ArchiveHandler]):
    """Register a new archive handler."""
    if handler not in _handlers:
        _handlers.append(handler)


def get_handlers() -> List[Type[ArchiveHandler]]:
    """Get a list of all registered archive handlers."""
    return _handlers.copy()
