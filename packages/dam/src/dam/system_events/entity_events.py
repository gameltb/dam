from dataclasses import dataclass
from typing import BinaryIO, Optional

from .base import BaseSystemEvent


@dataclass
class NewEntityCreatedEvent(BaseSystemEvent):
    """
    An event that is triggered when a new entity is created by a command.
    This is useful for recursive processing.
    """

    entity_id: int
    file_stream: Optional[BinaryIO] = None
    filename: Optional[str] = None
