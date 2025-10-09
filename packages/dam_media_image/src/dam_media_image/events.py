"""Image-related events for the DAM system."""

from dataclasses import dataclass

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


@dataclass
class ImageAssetDetected(BaseEvent):
    """Fired when a stored file is identified as an image."""

    entity: Entity
    file_id: int
