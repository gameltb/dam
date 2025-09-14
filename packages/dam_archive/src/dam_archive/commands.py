from dataclasses import dataclass
from typing import List, Optional

from dam.core.commands import BaseCommand


@dataclass
class UnbindSplitArchiveCommand(BaseCommand[None]):
    """
    A command to unbind a split archive by deleting its manifest and part info.
    """

    master_entity_id: int


@dataclass
class CreateMasterArchiveCommand(BaseCommand[None]):
    """
    A command to manually create a master entity for a split archive.
    """

    name: str
    part_entity_ids: List[int]


@dataclass
class DiscoverAndBindCommand(BaseCommand[None]):
    """
    A command to discover and bind split archives from a list of paths.
    """

    paths: List[str]


@dataclass
class SetArchivePasswordCommand(BaseCommand[None]):
    """
    A command to set the password for an archive.
    """

    entity_id: int
    password: str


@dataclass
class IngestArchiveMembersCommand(BaseCommand[None]):
    """
    A command to ingest members from an archive asset into the ECS world.
    This command returns a stream of events.
    """

    entity_id: int
    passwords: Optional[List[str]] = None


@dataclass
class TagArchivePartCommand(BaseCommand[None]):
    """
    A command to tag a file as a potential split archive part.
    """

    entity_id: int


@dataclass
class ClearArchiveComponentsCommand(BaseCommand[None]):
    """
    A command to clear archive-related components from an entity.
    """

    entity_id: int
