from dam.core.operations import AssetOperation

from .commands import (
    CheckPSPMetadataCommand,
    ExtractPSPMetadataCommand,
    RemovePSPMetadataCommand,
)

extract_psp_metadata_operation = AssetOperation(
    name="extract-psp-metadata",
    description="Extracts metadata from PSP ISO files.",
    add_command_class=ExtractPSPMetadataCommand,
    check_command_class=CheckPSPMetadataCommand,
    remove_command_class=RemovePSPMetadataCommand,
)
