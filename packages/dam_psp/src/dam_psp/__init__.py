"""Defines the plugin for PSP-related functionality in the DAM system."""

from __future__ import annotations

from dam.commands.asset_commands import GetAssetStreamCommand
from dam.core.plugin import Plugin
from dam.core.world import World
from dam.traits.asset_operation import AssetOperationTrait
from dam.traits.identifier import TraitImplementationIdentifier
from dam.traits.traits import TraitImplementation

from . import psp_iso_functions
from .commands import (
    CheckCsoIngestionCommand,
    CheckPSPMetadataCommand,
    ClearCsoIngestionCommand,
    ExtractPSPMetadataCommand,
    IngestCsoCommand,
    ReissueVirtualIsoEventCommand,
    RemovePSPMetadataCommand,
)
from .models import CsoDecompressionComponent, PspMetadataComponent
from .settings import PspSettingsComponent, PspSettingsModel
from .systems import (
    check_cso_ingestion_handler,
    check_psp_metadata_handler,
    clear_cso_ingestion_handler,
    get_virtual_iso_asset_stream_handler,
    ingest_cso_handler,
    psp_iso_metadata_extraction_command_handler_system,
    reissue_virtual_iso_event_handler,
    remove_psp_metadata_handler,
)


class PspPlugin(Plugin):
    """A plugin for handling PSP ISOs."""

    Settings = PspSettingsModel
    SettingsComponent = PspSettingsComponent

    def build(self, world: World) -> None:
        """Build the PSP plugin by registering its systems and operations."""
        world.register_system(
            psp_iso_metadata_extraction_command_handler_system,
            command_type=ExtractPSPMetadataCommand,
        )
        world.register_system(check_psp_metadata_handler, command_type=CheckPSPMetadataCommand)
        world.register_system(remove_psp_metadata_handler, command_type=RemovePSPMetadataCommand)
        world.register_system(ingest_cso_handler, command_type=IngestCsoCommand)
        world.register_system(check_cso_ingestion_handler, command_type=CheckCsoIngestionCommand)
        world.register_system(clear_cso_ingestion_handler, command_type=ClearCsoIngestionCommand)
        world.register_system(get_virtual_iso_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(reissue_virtual_iso_event_handler, command_type=ReissueVirtualIsoEventCommand)

        # Register Asset Operations
        extract_psp_metadata_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: psp_iso_metadata_extraction_command_handler_system,
                AssetOperationTrait.Check: check_psp_metadata_handler,
                AssetOperationTrait.Remove: remove_psp_metadata_handler,
            },
            identifier=TraitImplementationIdentifier.from_string(
                "asset.operation.extract_psp_metadata|PspMetadataComponent"
            ),
            name="extract-psp-metadata",
            description="Extracts metadata from PSP ISO files.",
        )
        world.trait_manager.register(PspMetadataComponent, extract_psp_metadata_implementation)

        decompress_cso_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: ingest_cso_handler,
                AssetOperationTrait.Check: check_cso_ingestion_handler,
                AssetOperationTrait.Remove: clear_cso_ingestion_handler,
            },
            identifier=TraitImplementationIdentifier.from_string(
                "asset.operation.decompress_cso|CsoDecompressionComponent"
            ),
            name="cso.decompress",
            description="Decompresses a CSO file into a virtual ISO.",
        )
        world.trait_manager.register(CsoDecompressionComponent, decompress_cso_implementation)


__all__ = ["PspPlugin", "psp_iso_functions"]
