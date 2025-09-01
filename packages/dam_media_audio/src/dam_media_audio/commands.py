import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dam.core.commands import BaseCommand
from dam.models.core.entity import Entity


@dataclass
class ExtractAudioMetadataCommand(BaseCommand[None]):
    entity: Entity

@dataclass
class AudioSearchCommand(BaseCommand[Optional[List[Tuple[Any, float, Any]]]]):
    """A command to perform a semantic search for audio."""
    query_audio_path: Path
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None
