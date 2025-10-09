"""Asset-related event definitions for the DAM system."""

from dataclasses import dataclass

from .base import BaseEvent


@dataclass
class AssetCreatedEvent(BaseEvent):
    """An event that is triggered when a new asset is created."""

    entity_id: int


@dataclass
class AssetUpdatedEvent(BaseEvent):
    """An event that is triggered when an asset is updated."""

    entity_id: int


@dataclass
class AssetDeletedEvent(BaseEvent):
    """An event that is triggered when an asset is deleted."""

    entity_id: int


@dataclass
class AssetReadyForMetadataExtractionEvent(BaseEvent):
    """An event that is triggered when a batch of assets is ready for metadata extraction."""

    entity_ids: list[int]
