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
class IngestAssetsCommand(BaseCommand[List[int]]):
    """A command to ingest new assets from a list of file paths."""

    file_paths: List[str]
    passwords: Optional[List[str]] = None
