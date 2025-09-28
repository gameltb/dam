from dataclasses import dataclass
from typing import Optional

from dam.core.types import StreamProvider

from .base import BaseSystemEvent


@dataclass
class NewEntityCreatedEvent(BaseSystemEvent):
    """
    An event that is triggered when a new entity is created by a command.
    This is useful for recursive processing.
    """

    entity_id: int
    stream_provider: Optional[StreamProvider] = None
    filename: Optional[str] = None
