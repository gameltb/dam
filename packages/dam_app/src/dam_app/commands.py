from dataclasses import dataclass

from dam.core.commands import BaseCommand
from dam.models.core.entity import Entity


@dataclass
class AutoTagEntityCommand(BaseCommand[None]):
    """A command to trigger auto-tagging for an entity."""

    entity: Entity


@dataclass
class ExtractMetadataCommand(BaseCommand[None]):
    """
    A command to trigger metadata extraction for an entity.
    """

    entity_id: int
    depth: int
