from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dam.commands.core import BaseCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class IngestWebAssetCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to ingest a new asset from a web source."""

    world_name: str
    website_identifier_url: str
    source_url: str
    metadata_payload: Optional[Dict[str, Any]] = None
    original_file_url: Optional[str] = None
    tags: Optional[List[str]] = None
