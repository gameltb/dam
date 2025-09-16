from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam.events import AssetReadyForMetadataExtractionEvent

if TYPE_CHECKING:
    from dam.core.world import World


from . import psp_iso_functions
from .commands import ExtractPspMetadataCommand
from .systems import (
    psp_iso_metadata_extraction_command_handler_system,
    psp_iso_metadata_extraction_event_handler_system,
)


class PspPlugin(Plugin):
    """
    A plugin for handling PSP ISOs.
    """

    def build(self, world: "World") -> None:
        """
        Builds the PSP plugin.
        """
        world.register_system(
            psp_iso_metadata_extraction_event_handler_system,
            event_type=AssetReadyForMetadataExtractionEvent,
        )
        world.register_system(
            psp_iso_metadata_extraction_command_handler_system,
            command_type=ExtractPspMetadataCommand,
        )


__all__ = ["PspPlugin", "psp_iso_functions"]
