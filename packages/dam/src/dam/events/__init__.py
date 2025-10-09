"""Core event definitions for the DAM system."""

from .asset_events import (
    AssetCreatedEvent,
    AssetDeletedEvent,
    AssetReadyForMetadataExtractionEvent,
    AssetUpdatedEvent,
)
from .base import BaseEvent

__all__ = [
    "AssetCreatedEvent",
    "AssetDeletedEvent",
    "AssetReadyForMetadataExtractionEvent",
    "AssetUpdatedEvent",
    "BaseEvent",
]
