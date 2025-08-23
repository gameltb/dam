from dataclasses import dataclass

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


import asyncio
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageAssetDetected(BaseEvent):
    """Fired when a stored file is identified as an image."""

    entity: Entity
    file_id: int


@dataclass
class FindSimilarImagesQuery(BaseEvent):
    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    world_name: str
    request_id: str
    result_future: Optional[asyncio.Future[List[Dict[str, Any]]]] = field(
        default=None, init=False, repr=False
    )
