"""Commands related to archive passwords."""

from dataclasses import dataclass

from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class SetArchivePasswordCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to set the password for an archive."""

    password: str


@dataclass
class CheckArchivePasswordCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an archive has a password set."""

    pass


@dataclass
class RemoveArchivePasswordCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to remove the password from an archive."""

    pass
