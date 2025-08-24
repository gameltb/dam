from dataclasses import dataclass
from typing import Optional

from dam.core.commands import BaseCommand


@dataclass
class IngestWebAssetCommand(BaseCommand):
    """A command to ingest a new asset from a web source."""

    world_name: str
    website_identifier_url: str
    source_url: str
    metadata_payload: Optional[dict] = None
    original_file_url: Optional[str] = None
    tags: Optional[list[str]] = None
