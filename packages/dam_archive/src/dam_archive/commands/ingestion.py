from dataclasses import dataclass
from typing import Dict, List, Optional

from dam.commands.analysis_commands import AnalysisCommand
from dam.commands.core import BaseCommand, EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class IngestArchiveCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to ingest members from an archive asset into the ECS world.
    This command returns a stream of events.
    """

    password: Optional[str] = None

    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        """
        Returns a dictionary of supported MIME types and file extensions for archives.
        """
        return {
            "mimetypes": [],
            "extensions": [".zip", ".rar", ".7z"],
        }


@dataclass
class TagArchivePartCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to tag a file as a potential split archive part.
    """

    entity_id: int


@dataclass
class CheckArchiveCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an entity has been processed as an archive."""

    pass


@dataclass
class ClearArchiveComponentsCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to clear archive-related components from an entity.
    """

    pass


@dataclass
class ReissueArchiveMemberEventsCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to re-issue NewEntityCreatedEvent events for all members of an existing archive.
    """

    pass
