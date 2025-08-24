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


