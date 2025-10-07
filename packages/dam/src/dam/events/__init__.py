"""Event definitions for the DAM system."""

from .asset_events import (
    AssetCreatedEvent,
    AssetDeletedEvent,
    AssetReadyForMetadataExtractionEvent,
    AssetUpdatedEvent,
)

__all__ = [
    "AssetCreatedEvent",
    "AssetDeletedEvent",
    "AssetReadyForMetadataExtractionEvent",
    "AssetUpdatedEvent",
]
