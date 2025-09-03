from dataclasses import dataclass
from dam.core.commands import BaseCommand

@dataclass
class SetArchivePasswordCommand(BaseCommand[None]):
    """
    A command to set the password for an archive.
    """
    entity_id: int
    password: str
