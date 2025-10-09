"""Defines commands for the `dam_fs` package."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dam.commands.core import BaseCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class AddFilePropertiesCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to add file properties to an entity."""

    entity_id: int
    original_filename: str
    size_bytes: int
    modified_at: datetime


@dataclass
class IngestReferenceCommand(BaseCommand[None, BaseSystemEvent]):
    """Command to ingest a file by reference (without copying)."""

    filepath_on_disk: Path
    original_filename: str
    size_bytes: int


@dataclass
class FindEntityByHashCommand(BaseCommand[dict[str, Any] | None, BaseSystemEvent]):
    """
    Command to find an entity by its content hash.

    The result is a dictionary with entity details or None, set on the future.
    """

    hash_value: str
    hash_type: str  # e.g., "sha256", "md5"
    request_id: str


@dataclass
class FindEntityByFilePropertiesCommand(BaseCommand[int | None, BaseSystemEvent]):
    """
    Command to find an entity by its file properties (path and modification time).

    The result is the entity ID or None.
    """

    file_path: str
    last_modified_at: datetime


@dataclass
class RegisterLocalFileCommand(BaseCommand[int | None, BaseSystemEvent]):
    """
    Command to register a local file, creating an entity if needed.

    Returns the entity ID, or None if the file does not exist.
    """

    file_path: Path


@dataclass
class StoreAssetsCommand(BaseCommand[None, BaseSystemEvent]):
    """Command to store assets based on a query."""

    query: str
