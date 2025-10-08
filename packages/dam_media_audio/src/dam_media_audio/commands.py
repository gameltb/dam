"""Defines commands for the `dam_media_audio` package."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dam.commands.core import BaseCommand
from dam.models.core.entity import Entity
from dam.system_events.base import BaseSystemEvent


@dataclass
class ExtractAudioMetadataCommand(BaseCommand[bool, BaseSystemEvent]):
    """A command to extract metadata from an audio file."""

    entity: Entity


@dataclass
class AudioSearchCommand(BaseCommand[list[tuple[Any, float, Any]] | None, BaseSystemEvent]):
    """A command to perform a semantic search for audio."""

    query_audio_path: Path
    request_id: str
    top_n: int = 10
    model_name: str | None = None
