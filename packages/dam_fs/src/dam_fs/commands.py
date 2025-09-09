from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dam.core.commands import BaseCommand


@dataclass
class AddFilePropertiesCommand(BaseCommand[None]):
    """
    A command to add file properties to an entity.
    """

    entity_id: int
    original_filename: str
    size_bytes: int


@dataclass
class IngestReferenceCommand(BaseCommand[None]):
    """Command to ingest a file by reference (without copying)."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int


@dataclass
class FindEntityByHashCommand(BaseCommand[Optional[Dict[str, Any]]]):
    """
    Command to find an entity by its content hash.
    The result is a dictionary with entity details or None, set on the future.
    """

    hash_value: str
    hash_type: str  # e.g., "sha256", "md5"
    request_id: str


@dataclass
class FindEntityByFilePropertiesCommand(BaseCommand[Optional[int]]):
    """
    Command to find an entity by its file properties (path and modification time).
    The result is the entity ID or None.
    """

    file_path: str
    file_modified_at: datetime


@dataclass
class RegisterLocalFileCommand(BaseCommand[int]):
    """
    Command to register a local file, creating an entity if needed.
    Returns the entity ID.
    """
    file_path: Path


@dataclass
class StoreAssetsCommand(BaseCommand[None]):
    """
    Command to store assets based on a query.
    """
    query: str
