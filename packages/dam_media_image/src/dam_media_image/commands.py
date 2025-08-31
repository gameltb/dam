import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dam.core.commands import BaseCommand


@dataclass
class FindSimilarImagesCommand(BaseCommand):
    """A command to find similar images based on perceptual hashes."""

    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    world_name: str
    request_id: str
    result_future: Optional[asyncio.Future[List[Dict[str, Any]]]] = field(default=None, init=False, repr=False)
