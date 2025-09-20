from dataclasses import dataclass
from typing import Dict, List

from dam.commands.analysis_commands import AnalysisCommand
from dam.system_events import BaseSystemEvent


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
