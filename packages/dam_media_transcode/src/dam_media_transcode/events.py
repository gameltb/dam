from dataclasses import dataclass
from pathlib import Path
from dam.core.events import BaseEvent


# --- Transcoding Events ---
@dataclass
class TranscodeJobRequested(BaseEvent):
    world_name: str
    source_entity_id: int
    profile_id: int  # This is the Entity ID of the TranscodeProfileComponent's entity
    priority: int = 100
    output_parent_dir: Path | None = None  # Optional: specify where the output file should be placed initially


@dataclass
class TranscodeJobCompleted(BaseEvent):
    job_id: int  # Corresponds to the ID in the TranscodeJobDB table
    world_name: str
    source_entity_id: int
    profile_id: int  # Entity ID of the profile used
    output_entity_id: int  # Entity ID of the newly created transcoded asset
    output_file_path: Path  # Path to the (potentially temporary) transcoded file


@dataclass
class TranscodeJobFailed(BaseEvent):
    job_id: int
    world_name: str
    source_entity_id: int
    profile_id: int
    error_message: str


# --- Evaluation Events ---
@dataclass
class StartEvaluationForTranscodedAsset(BaseEvent):
    world_name: str
    evaluation_run_id: int  # Entity ID of the EvaluationRun concept
    transcoded_asset_id: int  # Entity ID of the asset that was transcoded (output of a transcode job)
