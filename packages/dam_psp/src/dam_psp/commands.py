from dataclasses import dataclass
from typing import BinaryIO, Optional

from dam.core.commands import AnalysisCommand


@dataclass
class ExtractPSPMetadataCommand(AnalysisCommand[None]):
    """
    A command to extract metadata from a PSP ISO file.
    """

    pass
