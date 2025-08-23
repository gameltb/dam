import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dam.core.events import BaseEvent


@dataclass
class AudioSearchQuery(BaseEvent):
    query_audio_path: Path  # Path to the local query audio file
    world_name: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
    # Result future will yield List[Tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]]
    # Using 'Any' for BaseSpecificAudioEmbeddingComponent to avoid model import here.
    result_future: Optional[asyncio.Future[List[Tuple[Any, float, Any]]]] = field(default=None, init=False, repr=False)
