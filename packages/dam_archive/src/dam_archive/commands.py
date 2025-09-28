from dataclasses import dataclass
from typing import Dict, List, Optional

from dam.commands.analysis_commands import AnalysisCommand
from dam.commands.core import BaseCommand, EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class UnbindSplitArchiveCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to unbind a split archive by deleting its manifest and part info.
    """

    master_entity_id: int


@dataclass
class CreateMasterArchiveCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to manually create a master entity for a split archive.
    """

    name: str
    part_entity_ids: List[int]


@dataclass
class DiscoverAndBindCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to discover and bind split archives from a list of paths.
    """

    paths: List[str]


@dataclass
class SetArchivePasswordCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to set the password for an archive.
    """

    password: str


@dataclass
class CheckArchivePasswordCommand(EntityCommand[bool, BaseSystemEvent]):
    """
    A command to check if an archive has a password set.
    """

    pass


@dataclass
class RemoveArchivePasswordCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to remove the password from an archive.
    """

    pass


@dataclass
class IngestArchiveCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to ingest members from an archive asset into the ECS world.
    This command returns a stream of events.
    """

    passwords: Optional[List[str]] = None

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
