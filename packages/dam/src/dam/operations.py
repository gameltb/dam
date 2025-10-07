"""Core asset operations for the DAM system."""

from dam.commands.asset_commands import (
    CheckContentMimeTypeCommand,
    RemoveContentMimeTypeCommand,
    SetMimeTypeCommand,
)
from dam.core.operations import AssetOperation


class SetMimeTypeOperation(AssetOperation):
    """Operation for setting the mime type of an asset."""

    name = "core.set_mime_type"
    description = "Sets the mime type for an asset."
    add_command_class = SetMimeTypeCommand
    check_command_class = CheckContentMimeTypeCommand
    remove_command_class = RemoveContentMimeTypeCommand
