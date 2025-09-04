from dataclasses import dataclass
from typing import BinaryIO

from dam.core.commands import BaseCommand
from dam.models.core.entity import Entity


@dataclass
class IngestAssetStreamCommand(BaseCommand):
    """A command to ingest a new asset from an in-memory stream."""

    entity: Entity
    file_content: BinaryIO
    original_filename: str
    world_name: str


@dataclass
class AutoTagEntityCommand(BaseCommand):
    """A command to trigger auto-tagging for an entity."""

    entity: Entity


@dataclass
class ExtractMetadataCommand(BaseCommand):
    """
    A command to trigger metadata extraction for an entity.
    """

    entity_id: int
