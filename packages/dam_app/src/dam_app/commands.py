"""Defines custom commands for the DAM application."""

from dataclasses import dataclass

from dam.commands.analysis_commands import AnalysisCommand
from dam.commands.core import BaseCommand, EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class AnalyzeEntityCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to analyze an entity."""

    pass


@dataclass
class AutoTagEntityCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to automatically tag an entity."""

    pass


@dataclass
class ExtractExifMetadataCommand(AnalysisCommand[None, BaseSystemEvent]):
    """A command to extract metadata from an entity."""

    pass


@dataclass
class CheckExifMetadataCommand(EntityCommand[bool, BaseSystemEvent]):
    """A command to check if an entity has extracted EXIF metadata."""

    pass


@dataclass
class RemoveExifMetadataCommand(EntityCommand[None, BaseSystemEvent]):
    """A command to remove the extracted EXIF metadata from an entity."""

    pass


@dataclass
class ExportDbCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to export the database to a file."""

    pass


@dataclass
class MigratePathsCommand(BaseCommand[None, BaseSystemEvent]):
    """A command to migrate existing file paths to the new path tree structure."""

    pass
