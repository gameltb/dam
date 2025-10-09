"""Defines the asset operations for the DAM application."""

from dam.core.operations import AssetOperation

from .commands import (
    CheckExifMetadataCommand,
    ExtractExifMetadataCommand,
    RemoveExifMetadataCommand,
)

extract_exif_operation = AssetOperation(
    name="extract-exif-metadata",
    description="Extracts EXIF metadata from image files.",
    add_command_class=ExtractExifMetadataCommand,
    check_command_class=CheckExifMetadataCommand,
    remove_command_class=RemoveExifMetadataCommand,
)
