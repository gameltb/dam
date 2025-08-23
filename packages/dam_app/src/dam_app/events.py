from dataclasses import dataclass
from typing import BinaryIO

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


@dataclass
class AssetStreamIngestionRequested(BaseEvent):
    """Fired when a new asset is being ingested from an in-memory stream."""

    entity: Entity
    file_content: BinaryIO
    original_filename: str
    world_name: str
