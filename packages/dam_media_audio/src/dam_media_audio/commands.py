from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dam.commands.core import BaseCommand
from dam.models.core.entity import Entity
from dam.system_events import BaseSystemEvent


@dataclass
class ExtractAudioMetadataCommand(BaseCommand[bool, BaseSystemEvent]):
    entity: Entity


@dataclass
class AudioSearchCommand(BaseCommand[Optional[List[Tuple[Any, float, Any]]], BaseSystemEvent]):
    """A command to perform a semantic search for audio."""

    query_audio_path: Path
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None
