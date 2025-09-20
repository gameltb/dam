from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dam.commands.core import BaseCommand
from dam.system_events import BaseSystemEvent


@dataclass
class FindSimilarImagesCommand(BaseCommand[Optional[List[Dict[str, Any]]], BaseSystemEvent]):
    """A command to find similar images based on perceptual hashes."""

    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    request_id: str
