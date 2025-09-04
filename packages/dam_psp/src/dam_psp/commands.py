from dataclasses import dataclass

from dam.core.commands import BaseCommand


@dataclass
class ExtractPSPMetadataCommand(BaseCommand[None]):
    """
    A command to extract metadata from a PSP ISO file.
    """

    entity_id: int
