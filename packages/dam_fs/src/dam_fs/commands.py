from asyncio import Future
from dataclasses import dataclass, field
from pathlib import Path

from dam.core.commands import BaseCommand


@dataclass
class IngestFileCommand(BaseCommand[None]):
    """Command to ingest a file by copying it into the asset storage."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int


@dataclass
class IngestReferenceCommand(BaseCommand[None]):
    """Command to ingest a file by reference (without copying)."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int


@dataclass
class FindEntityByHashCommand(BaseCommand[dict | None]):
    """
    Command to find an entity by its content hash.
    The result is a dictionary with entity details or None, set on the future.
    """

    hash_value: str
    hash_type: str  # e.g., "sha256", "md5"
    request_id: str
