"""Defines asset operations related to PSP files."""
from dam.core.operations import AssetOperation

from .commands import (
    CheckCsoIngestionCommand,
    CheckPSPMetadataCommand,
    ClearCsoIngestionCommand,
    ExtractPSPMetadataCommand,
    IngestCsoCommand,
    ReissueVirtualIsoEventCommand,
    RemovePSPMetadataCommand,
)

extract_psp_metadata_operation = AssetOperation(
    name="extract-psp-metadata",
    description="Extracts metadata from PSP ISO files.",
    add_command_class=ExtractPSPMetadataCommand,
    check_command_class=CheckPSPMetadataCommand,
    remove_command_class=RemovePSPMetadataCommand,
)

decompress_cso_operation = AssetOperation(
    name="cso.decompress",
    description="Decompresses a CSO file into a virtual ISO.",
    add_command_class=IngestCsoCommand,
    check_command_class=CheckCsoIngestionCommand,
    reprocess_derived_command_class=ReissueVirtualIsoEventCommand,
    remove_command_class=ClearCsoIngestionCommand,
)
