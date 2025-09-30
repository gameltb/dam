from dam.core.operations import AssetOperation

from .commands import (
    CheckCsoIngestionCommand,
    CheckPSPMetadataCommand,
    ClearCsoIngestionCommand,
    ExtractPSPMetadataCommand,
    IngestCsoCommand,
    RemovePSPMetadataCommand,
)

extract_psp_metadata_operation = AssetOperation(
    name="extract-psp-metadata",
    description="Extracts metadata from PSP ISO files.",
    add_command_class=ExtractPSPMetadataCommand,
    check_command_class=CheckPSPMetadataCommand,
    remove_command_class=RemovePSPMetadataCommand,
)

ingest_cso_operation = AssetOperation(
    name="ingest-cso",
    description="Ingests a CSO file, creating a virtual ISO entity.",
    add_command_class=IngestCsoCommand,
    check_command_class=CheckCsoIngestionCommand,
    remove_command_class=ClearCsoIngestionCommand,
)
