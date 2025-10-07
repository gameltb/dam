from dataclasses import dataclass

from dam.commands.core import BaseCommand, EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class UnbindSplitArchiveCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to unbind a split archive by deleting its manifest and part info.
    When given the entity_id of a master archive, it unbinds it.
    When given the entity_id of a part, it finds the master and unbinds it.
    """

    pass


@dataclass
class CreateMasterArchiveCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to manually create a master entity for a split archive."""

    name: str
    part_entity_ids: list[int]


@dataclass
class BindSplitArchiveCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command that attempts to discover and bind a split archive group
    starting from the given entity.
    """

    pass


@dataclass
class CheckSplitArchiveBindingCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an entity is part of a fully bound split archive."""

    pass
