"""Defines commands related to PSP ISO and CSO file handling."""
from dataclasses import dataclass

from dam.commands.analysis_commands import AnalysisCommand
from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class CheckPSPMetadataCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an entity has PSP metadata."""


@dataclass
class RemovePSPMetadataCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to remove PSP metadata from an entity."""


@dataclass
class ExtractPSPMetadataCommand(AnalysisCommand[None, BaseSystemEvent]):
    """A command to extract metadata from a PSP ISO file."""

    @classmethod
    def get_supported_types(cls) -> dict[str, list[str]]:
        """Return a dictionary of supported MIME types and file extensions for PSP ISOs."""
        return {
            "mimetypes": ["application/x-iso9660-image"],
            "extensions": [".iso"],
        }


@dataclass
class CheckCsoIngestionCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if a CSO file has been ingested."""


@dataclass
class ClearCsoIngestionCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to clear the ingestion data from a CSO file."""


@dataclass
class IngestCsoCommand(AnalysisCommand[None, BaseSystemEvent]):
    """A command to ingest a CSO file, decompress it, and create a virtual ISO entity."""

    @classmethod
    def get_supported_types(cls) -> dict[str, list[str]]:
        """Return a dictionary of supported MIME types and file extensions for CSO files."""
        return {
            "mimetypes": ["application/x-ciso"],
            "extensions": [".cso"],
        }


@dataclass
class ReissueVirtualIsoEventCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to re-issue a NewEntityCreatedEvent for the virtual ISO derived from a CSO file."""
