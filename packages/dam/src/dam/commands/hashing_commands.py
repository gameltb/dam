from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Set

from dam.commands.core import BaseCommand
from dam.system_events import BaseSystemEvent
from dam.utils.hash_utils import HashAlgorithm


@dataclass
class AddHashesFromStreamCommand(BaseCommand[None, BaseSystemEvent]):
    """Command to calculate and add multiple hash components to an entity from a stream."""

    entity_id: int
    stream: BinaryIO
    algorithms: Set[HashAlgorithm]
