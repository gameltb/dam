"""Core components for asset operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class AssetOperation:
    """A data class that defines a specific operation that can be performed on an asset."""

    # The unique name for the operation, e.g., "exif.extract"
    name: str
    description: str
    add_command_class: type[EntityCommand[Any, Any]]
    check_command_class: type[EntityCommand[bool, BaseSystemEvent]] | None = None
    remove_command_class: type[EntityCommand[None, BaseSystemEvent]] | None = None
    reprocess_derived_command_class: type[EntityCommand[Any, Any]] | None = None

    def get_supported_types(self) -> dict[str, list[str]]:
        """
        Return a dictionary of supported MIME types and file extensions.

        This is delegated from the add_command.
        """
        if hasattr(self.add_command_class, "get_supported_types"):
            # The ignore is needed because the base EntityCommand doesn't have this method,
            # but the commands that support this do.
            return self.add_command_class.get_supported_types()  # type: ignore
        return {"mimetypes": [], "extensions": []}
