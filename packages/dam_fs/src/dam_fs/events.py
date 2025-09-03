from dataclasses import dataclass
from pathlib import Path
from typing import List

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


@dataclass
class AssetsReadyForMetadataExtraction(BaseEvent):
    """
    An event that is triggered when a batch of assets is ready for metadata extraction.
    """

    entity_ids: List[int]


@dataclass
class FileStored(BaseEvent):
    """Fired after a file has been saved to the Content-Addressable Storage."""

    entity: Entity
    file_id: int
    file_path: Path
