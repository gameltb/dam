from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from ..core.types import StreamProvider
from ..system_events.base import BaseSystemEvent
from ..utils.hash_utils import HashAlgorithm
from .core import BaseCommand


@dataclass
class AddHashesFromStreamCommand(BaseCommand[None, BaseSystemEvent]):
    """Command to calculate and add multiple hash components to an entity from a stream."""

    entity_id: int
    stream_provider: StreamProvider
    algorithms: Set[HashAlgorithm]
