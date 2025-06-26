from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

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

# To allow __init__.py in dam/core to import these
__all__ = [
    "BaseEvent",
    "AssetFileIngestionRequested",
    "AssetReferenceIngestionRequested",
]
