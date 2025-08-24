import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dam.core.commands import BaseCommand


from dam.models.core.entity import Entity


@dataclass
class ProcessAudioCommand(BaseCommand):
    """A command to trigger audio processing for an entity."""

    entity: Entity


@dataclass
class AudioSearchCommand(BaseCommand):
    """A command to perform an audio similarity search."""

    query_audio_path: Path  # Path to the local query audio file
    world_name: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
    result_future: Optional[asyncio.Future[List[Tuple[Any, float, Any]]]] = field(
        default=None, init=False, repr=False
    )
