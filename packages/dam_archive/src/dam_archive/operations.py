from dam.core.operations import AssetOperation

from .commands import (
    CheckArchiveCommand,
    CheckArchivePasswordCommand,
    ClearArchiveComponentsCommand,
    IngestArchiveCommand,
    RemoveArchivePasswordCommand,
    SetArchivePasswordCommand,
)

ingest_archive_operation = AssetOperation(
    name="ingest-archive",
    description="Ingests members from an archive file.",
    add_command_class=IngestArchiveCommand,
    check_command_class=CheckArchiveCommand,
    remove_command_class=ClearArchiveComponentsCommand,
)

set_archive_password_operation = AssetOperation(
    name="set-archive-password",
    description="Sets the password for an archive.",
    add_command_class=SetArchivePasswordCommand,
    check_command_class=CheckArchivePasswordCommand,
    remove_command_class=RemoveArchivePasswordCommand,
)
