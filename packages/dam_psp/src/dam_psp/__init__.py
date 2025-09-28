from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

if TYPE_CHECKING:
    from dam.core.world import World


from . import psp_iso_functions
from .commands import (
    CheckPSPMetadataCommand,
    ExtractPSPMetadataCommand,
    RemovePSPMetadataCommand,
)
from .operations import extract_psp_metadata_operation
from .systems import (
    check_psp_metadata_handler,
    psp_iso_metadata_extraction_command_handler_system,
    remove_psp_metadata_handler,
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
            psp_iso_metadata_extraction_command_handler_system,
            command_type=ExtractPSPMetadataCommand,
        )
        world.register_system(check_psp_metadata_handler, command_type=CheckPSPMetadataCommand)
        world.register_system(remove_psp_metadata_handler, command_type=RemovePSPMetadataCommand)

        # Register Asset Operations
        world.register_asset_operation(extract_psp_metadata_operation)


__all__ = ["PspPlugin", "psp_iso_functions", "extract_psp_metadata_operation"]
