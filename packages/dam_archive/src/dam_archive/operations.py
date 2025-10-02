from dam.core.operations import AssetOperation

from .commands.ingestion import (
    CheckArchiveCommand,
    ClearArchiveComponentsCommand,
    IngestArchiveCommand,
    ReissueArchiveMemberEventsCommand,
)
from .commands.password import (
    CheckArchivePasswordCommand,
    RemoveArchivePasswordCommand,
    SetArchivePasswordCommand,
)
from .commands.split_archives import (
    BindSplitArchiveCommand,
    CheckSplitArchiveBindingCommand,
    UnbindSplitArchiveCommand,
)

bind_split_archive_operation = AssetOperation(
    name="archive.bind-split-archive",
    description="Finds and binds all parts of a split archive into a single master entity.",
    add_command_class=BindSplitArchiveCommand,
    check_command_class=CheckSplitArchiveBindingCommand,
    remove_command_class=UnbindSplitArchiveCommand,
)

ingest_archive_operation = AssetOperation(
    name="archive.ingest",
    description="Ingests members from an archive file.",
    add_command_class=IngestArchiveCommand,
    check_command_class=CheckArchiveCommand,
    remove_command_class=ClearArchiveComponentsCommand,
    reprocess_derived_command_class=ReissueArchiveMemberEventsCommand,
)

set_archive_password_operation = AssetOperation(
    name="archive.set-password",
    description="Sets the password for an archive.",
    add_command_class=SetArchivePasswordCommand,
    check_command_class=CheckArchivePasswordCommand,
    remove_command_class=RemoveArchivePasswordCommand,
)
