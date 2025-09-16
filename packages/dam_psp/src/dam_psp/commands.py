from dataclasses import dataclass
from typing import BinaryIO, Optional

from dam.core.commands import BaseCommand


@dataclass
class ExtractPspMetadataCommand(BaseCommand[None]):
    """
    A command to extract metadata from a PSP ISO file.
    """

    entity_id: int
    depth: int
    stream: Optional[BinaryIO] = None
