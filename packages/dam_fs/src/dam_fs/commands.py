from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from dam.core.commands import BaseCommand
from dam.models.core.entity import Entity


@dataclass
class AddFilePropertiesCommand(BaseCommand[None]):
    """
    A command to add file properties to an entity.
    """
    entity_id: int
    original_filename: str
    size_bytes: int


@dataclass
class GetAssetStreamCommand(BaseCommand[BinaryIO]):
    """A command to get a readable stream for an asset."""

    entity_id: int


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
