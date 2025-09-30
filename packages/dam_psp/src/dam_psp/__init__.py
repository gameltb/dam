from __future__ import annotations

from dam.commands.asset_commands import GetAssetStreamCommand
from dam.core.plugin import Plugin
from dam.core.world import World

from . import psp_iso_functions
from .commands import (
    CheckCsoIngestionCommand,
    CheckPSPMetadataCommand,
    ClearCsoIngestionCommand,
    ExtractPSPMetadataCommand,
    IngestCsoCommand,
    RemovePSPMetadataCommand,
)
from .operations import extract_psp_metadata_operation, ingest_cso_operation
from .systems import (
    check_cso_ingestion_handler,
    check_psp_metadata_handler,
    clear_cso_ingestion_handler,
    get_virtual_iso_asset_stream_handler,
    ingest_cso_handler,
    psp_iso_metadata_extraction_command_handler_system,
    remove_psp_metadata_handler,
)


class PspPlugin(Plugin):
    """
    A plugin for handling PSP ISOs.
    """

    def build(self, world: World) -> None:
        """
        Builds the PSP plugin.
        """
        world.register_system(
            psp_iso_metadata_extraction_command_handler_system,
            command_type=ExtractPSPMetadataCommand,
        )
        world.register_system(check_psp_metadata_handler, command_type=CheckPSPMetadataCommand)
        world.register_system(remove_psp_metadata_handler, command_type=RemovePSPMetadataCommand)
        world.register_system(
            ingest_cso_handler,
            command_type=IngestCsoCommand
        )
        world.register_system(check_cso_ingestion_handler, command_type=CheckCsoIngestionCommand)
        world.register_system(clear_cso_ingestion_handler, command_type=ClearCsoIngestionCommand)
        world.register_system(get_virtual_iso_asset_stream_handler, command_type=GetAssetStreamCommand)

        # Register Asset Operations
        world.register_asset_operation(extract_psp_metadata_operation)
        world.register_asset_operation(ingest_cso_operation)


__all__ = ["PspPlugin", "psp_iso_functions", "extract_psp_metadata_operation"]
