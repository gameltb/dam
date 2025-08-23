from dataclasses import dataclass
from pathlib import Path

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


import asyncio
from typing import Any, Dict, Optional


@dataclass
class FileStored(BaseEvent):
    """Fired after a file has been saved to the Content-Addressable Storage."""

    entity: Entity
    file_id: int
    file_path: Path


@dataclass
class AssetFileIngestionRequested(BaseEvent):
    filepath_on_disk: Path
    original_filename: str
    size_bytes: int
    world_name: str


@dataclass
class AssetReferenceIngestionRequested(BaseEvent):
    filepath_on_disk: Path
    original_filename: str
    size_bytes: int
    world_name: str


from dataclasses import field

@dataclass
class FindEntityByHashQuery(BaseEvent):
    hash_value: str
    world_name: str
    request_id: str
    hash_type: str = "sha256"
    result_future: Optional[asyncio.Future[Optional[Dict[str, Any]]]] = field(default=None, init=False, repr=False)
