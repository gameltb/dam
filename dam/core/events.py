from dataclasses import dataclass
from pathlib import Path

# Using dataclasses for simplicity, similar to Bevy events.


@dataclass
class BaseEvent:
    """Base class for all events, providing a common structure if needed."""

    pass


@dataclass
class AssetFileIngestionRequested(BaseEvent):
    """
    Event dispatched when a new asset file needs to be ingested by copying.
    Corresponds to the logic previously in asset_service.add_asset_file.
    """

    filepath_on_disk: Path
    original_filename: str
    mime_type: str
    size_bytes: int
    world_name: str  # Target world for this asset

    # To store the result of the operation for the dispatcher or caller to potentially access
    # This is optional and depends on how we want to get results back from event handlers.
    # For now, systems might update the DB and log, direct return might not be needed from the event itself.
    # entity_id: Optional[int] = field(default=None, init=False)
    # created_new: Optional[bool] = field(default=None, init=False)


@dataclass
class AssetReferenceIngestionRequested(BaseEvent):
    """
    Event dispatched when a new asset needs to be ingested by reference.
    Corresponds to the logic previously in asset_service.add_asset_reference.
    """

    filepath_on_disk: Path
    original_filename: str
    mime_type: str
    size_bytes: int
    world_name: str  # Target world for this asset

    # entity_id: Optional[int] = field(default=None, init=False)
    # created_new: Optional[bool] = field(default=None, init=False)


# Example of a potential event that could be dispatched *by* an ingestion system
# after an asset is successfully processed, if other systems need to react to that.
# For now, the `NeedsMetadataExtractionComponent` marker serves a similar purpose.
# @dataclass
# class AssetIngested(BaseEvent):
#     entity_id: int
#     world_name: str
#     ingestion_type: str # e.g., "file_copy", "reference"
#     original_filename: str

# Placeholder for a system to return results if needed.
# For now, systems will modify the DB directly.
# If direct feedback to the caller of event dispatch is needed, a different pattern might be used.
# For example, the dispatch method could return a future or a result object.
# For Bevy-like events, typically events are fire-and-forget, and systems react.
# Results are observed through changes in World state (Components, Resources).

# We might also need an event to trigger the metadata extraction stage,
# or the current mechanism of adding a marker component is sufficient.
# Let's stick to marker components for now as it's already in place.


# --- Events for Query Operations ---
# These events will carry a unique request_id to potentially correlate results
# if results are posted back as another event or stored in a temporary resource.


@dataclass
class FindEntityByHashQuery(BaseEvent):
    """Event to request finding an entity by its content hash."""

    hash_value: str
    world_name: str  # Target world for this query
    request_id: str  # Unique ID for this query request
    hash_type: str = "sha256"  # Moved to the end as it has a default

    # For carrying results if the system modifies the event or posts a new one
    # result_entity_id: Optional[int] = field(default=None, init=False)


@dataclass
class FindSimilarImagesQuery(BaseEvent):
    """Event to request finding similar images."""

    image_path: Path  # Path to the query image on a system accessible to the DAM
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    world_name: str  # Target world for this query
    request_id: str  # Unique ID for this query request

    # For carrying results
    # result_similar_entities_info: Optional[list[dict]] = field(default=None, init=False)


# To allow __init__.py in dam/core to import these
__all__ = [
    "BaseEvent",
    "AssetFileIngestionRequested",
    "AssetReferenceIngestionRequested",
    "FindEntityByHashQuery",
    "FindSimilarImagesQuery",
]
