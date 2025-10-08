"""Defines commands for the `dam_source` package."""

from dataclasses import dataclass
from typing import Any

from dam.commands.core import BaseCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class IngestWebAssetCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to ingest a new asset from a web source."""

    world_name: str
    website_identifier_url: str
    source_url: str
    metadata_payload: dict[str, Any] | None = None
    original_file_url: str | None = None
    tags: list[str] | None = None
