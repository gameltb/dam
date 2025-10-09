"""Defines event models for the `dam_fs` package."""

from dataclasses import dataclass
from pathlib import Path

from dam.events import BaseEvent
from dam.models.core.entity import Entity


@dataclass
class FileStored(BaseEvent):
    """Fired after a file has been saved to the Content-Addressable Storage."""

    entity: Entity
    file_id: int
    file_path: Path
