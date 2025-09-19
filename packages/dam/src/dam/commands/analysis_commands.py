from dataclasses import dataclass

from dam.core.commands import AnalysisCommand
from dam.system_events import BaseSystemEvent


@dataclass
class AutoSetMimeTypeCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to automatically set the mime type for an asset.
    """

    entity_id: int
