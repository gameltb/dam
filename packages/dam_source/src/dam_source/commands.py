from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from dam.core.commands import BaseCommand


@dataclass
class IngestWebAssetCommand(BaseCommand[None]):
    """A command to ingest a new asset from a web source."""

    world_name: str
    website_identifier_url: str
    source_url: str
    metadata_payload: Optional[Dict[str, Any]] = None
    original_file_url: Optional[str] = None
    tags: Optional[List[str]] = None
