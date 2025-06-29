import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Using dataclasses for simplicity, similar to Bevy events.

@dataclass
class BaseEvent:
    """Base class for all events, providing a common structure if needed."""
    pass

@dataclass
class AssetFileIngestionRequested(BaseEvent):
    filepath_on_disk: Path
    original_filename: str
    mime_type: str
    size_bytes: int
    world_name: str

@dataclass
class AssetReferenceIngestionRequested(BaseEvent):
    filepath_on_disk: Path
    original_filename: str
    mime_type: str
    size_bytes: int
    world_name: str

@dataclass
class FindEntityByHashQuery(BaseEvent):
    hash_value: str
    world_name: str
    request_id: str
    hash_type: str = "sha256"
    result_future: Optional[asyncio.Future[Optional[Dict[str, Any]]]] = field(default=None, init=False, repr=False)

@dataclass
class FindSimilarImagesQuery(BaseEvent):
    image_path: Path
    phash_threshold: int
    ahash_threshold: int
    dhash_threshold: int
    world_name: str
    request_id: str
    result_future: Optional[asyncio.Future[List[Dict[str, Any]]]] = field(default=None, init=False, repr=False) # Corrected here

@dataclass
class WebAssetIngestionRequested(BaseEvent):
    world_name: str
    website_identifier_url: str
    source_url: str
    metadata_payload: Optional[dict] = None
    original_file_url: Optional[str] = None
    tags: Optional[list[str]] = None

# --- Transcoding Events (Moved from dam.services.transcode_service) ---
@dataclass
class TranscodeJobRequested(BaseEvent):
    world_name: str
    source_entity_id: int
    profile_id: int # This is the Entity ID of the TranscodeProfileComponent's entity
    priority: int = 100
    output_parent_dir: Optional[Path] = None # Optional: specify where the output file should be placed initially

@dataclass
class TranscodeJobCompleted(BaseEvent):
    job_id: int # Corresponds to the ID in the TranscodeJobDB table
    world_name: str
    source_entity_id: int
    profile_id: int # Entity ID of the profile used
    output_entity_id: int # Entity ID of the newly created transcoded asset
    output_file_path: Path # Path to the (potentially temporary) transcoded file

@dataclass
class TranscodeJobFailed(BaseEvent):
    job_id: int
    world_name: str
    source_entity_id: int
    profile_id: int
    error_message: str

# --- Evaluation Events (Moved from dam.services.transcode_service) ---
@dataclass
class StartEvaluationForTranscodedAsset(BaseEvent):
    world_name: str
    evaluation_run_id: int # Entity ID of the EvaluationRun concept
    transcoded_asset_id: int # Entity ID of the asset that was transcoded (output of a transcode job)

__all__ = [
    "BaseEvent",
    "AssetFileIngestionRequested",
    "AssetReferenceIngestionRequested",
    "FindEntityByHashQuery",
    "FindSimilarImagesQuery",
    "WebAssetIngestionRequested",
    "TranscodeJobRequested",
    "TranscodeJobCompleted",
    "TranscodeJobFailed",
    "StartEvaluationForTranscodedAsset",
]
