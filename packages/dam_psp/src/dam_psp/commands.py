from dataclasses import dataclass
from typing import Dict, List

from dam.commands.analysis_commands import AnalysisCommand
from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class CheckPSPMetadataCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an entity has PSP metadata."""

    pass


@dataclass
class RemovePSPMetadataCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to remove PSP metadata from an entity."""

    pass


@dataclass
class ExtractPSPMetadataCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to extract metadata from a PSP ISO file.
    """

    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        """
        Returns a dictionary of supported MIME types and file extensions for PSP ISOs.
        """
        return {
            "mimetypes": ["application/x-iso9660-image"],
            "extensions": [".iso"],
        }
