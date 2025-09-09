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
class ProcessArchiveCommand(BaseCommand[None]):
    """
    A command to process an archive asset and ingest its members.
    """
    entity_id: int
    passwords: Optional[List[str]] = None
