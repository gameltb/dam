from dataclasses import dataclass
from typing import List, Optional

from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class PathSibling:
    """
    Represents a single "sibling" entity found by a path discovery command.
    """

    entity_id: int
    path: str


@dataclass
class DiscoverPathSiblingsCommand(EntityCommand[Optional[List[PathSibling]], BaseSystemEvent]):
    """
    A command to discover "sibling" entities for a given entity based on path.

    "Siblings" are entities that share a common path context, such as being in
    the same filesystem directory or the same directory within an archive.

    Different plugins can provide system handlers for this command. The command
    executor will return the first non-None list of PathSibling objects
    returned by a handler.
    """

    pass