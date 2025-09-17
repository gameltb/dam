from dataclasses import dataclass

from dam.core.commands import AnalysisCommand, BaseCommand
from dam.models.core.entity import Entity


@dataclass
class AutoTagEntityCommand(BaseCommand[None]):
    """A command to trigger auto-tagging for an entity."""

    entity: Entity


@dataclass
class ExtractExifMetadataCommand(AnalysisCommand[None]):
    """
    A command to trigger metadata extraction for an entity.
    Inherits entity_id, depth, and stream from AnalysisCommand.
    """

    pass
