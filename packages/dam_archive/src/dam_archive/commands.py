from dataclasses import dataclass
from typing import List, Optional

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
