import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dam.core.commands import BaseCommand


@dataclass
class FindSimilarImagesCommand(BaseCommand[Optional[List[Dict[str, Any]]]]):
    """A command to find similar images based on perceptual hashes."""

    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    request_id: str
