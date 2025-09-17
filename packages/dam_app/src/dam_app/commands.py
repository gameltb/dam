from dataclasses import dataclass

from dam.core.commands import AnalysisCommand, BaseCommand
from dam.models.core.entity import Entity
from dam.system_events import BaseSystemEvent


@dataclass
class AutoTagEntityCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to trigger auto-tagging for an entity."""

    entity: Entity


@dataclass
class ExtractExifMetadataCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to trigger metadata extraction for an entity.
    Inherits entity_id, depth, and stream from AnalysisCommand.
    """

    pass
