from dataclasses import dataclass

from dam.core.commands import AnalysisCommand
from dam.system_events import BaseSystemEvent


@dataclass
class ExtractPSPMetadataCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to extract metadata from a PSP ISO file.
    """

    pass
