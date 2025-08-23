from dataclasses import dataclass

from dam.core.events import BaseEvent
from dam.models.core.entity import Entity


@dataclass
class PspIsoAssetDetected(BaseEvent):
    """Fired when a stored file is identified as a PSP ISO."""

    entity: Entity
    file_id: int
