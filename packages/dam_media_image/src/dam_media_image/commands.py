"""Image-related commands for the DAM system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dam.commands.core import BaseCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class FindSimilarImagesCommand(BaseCommand[list[dict[str, Any]] | None, BaseSystemEvent]):
    """A command to find similar images based on perceptual hashes."""

    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    request_id: str
