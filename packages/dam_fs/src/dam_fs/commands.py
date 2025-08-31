import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dam.core.commands import BaseCommand


@dataclass
class IngestFileCommand(BaseCommand):
    """A command to ingest a new asset from a file path."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int
    world_name: str


@dataclass
class IngestReferenceCommand(BaseCommand):
    """A command to ingest a new asset by reference (symlink)."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int
    world_name: str


@dataclass
class FindEntityByHashCommand(BaseCommand):
    """A command to find an entity by its hash."""

    hash_value: str
    world_name: str
    request_id: str
    hash_type: str = "sha256"
    result_future: Optional[asyncio.Future[Optional[Dict[str, Any]]]] = field(default=None, init=False, repr=False)
