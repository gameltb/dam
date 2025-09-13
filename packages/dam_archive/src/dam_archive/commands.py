from dataclasses import dataclass, field
from typing import Callable, List, Optional

from dam.core.commands import BaseCommand


@dataclass
class SetArchivePasswordCommand(BaseCommand[None]):
    """
    A command to set the password for an archive.
    """

    entity_id: int
    password: str


@dataclass
class ExtractArchiveMembersCommand(BaseCommand[None]):
    """
    A command to extract members from an archive asset and ingest them.
    """

    entity_id: int
    passwords: Optional[List[str]] = None
    init_progress_callback: Optional[Callable[[int], None]] = field(default=None, repr=False)
    update_progress_callback: Optional[Callable[[int], None]] = field(default=None, repr=False)
    error_callback: Optional[Callable[[str, Exception], bool]] = field(default=None, repr=False)


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
