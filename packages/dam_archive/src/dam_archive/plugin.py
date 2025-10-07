"""Defines the archive plugin."""

from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
)
from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.plugin import Plugin
from dam.core.world import World

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
    CreateMasterArchiveCommand,
    UnbindSplitArchiveCommand,
)
from .operations import (
    bind_split_archive_operation,
    ingest_archive_operation,
    set_archive_password_operation,
)
from .systems.discovery import discover_archive_path_siblings_handler
from .systems.ingestion import (
    check_archive_handler,
    clear_archive_components_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    ingest_archive_members_handler,
    reissue_archive_member_events_handler,
)
from .systems.password import (
    check_archive_password_handler,
    remove_archive_password_handler,
    set_archive_password_handler,
)
from .systems.split_archives import (
    bind_split_archive_handler,
    check_split_archive_binding_handler,
    create_master_archive_handler,
    unbind_split_archive_handler,
)


class ArchivePlugin(Plugin):
    """A plugin for handling archives."""

    def build(self, world: World) -> None:
        """Build the archive plugin."""
        # Command Handlers
        world.register_system(create_master_archive_handler, command_type=CreateMasterArchiveCommand)
        world.register_system(unbind_split_archive_handler, command_type=UnbindSplitArchiveCommand)
        world.register_system(get_archive_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(set_archive_password_handler, command_type=SetArchivePasswordCommand)
        world.register_system(get_archive_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
        world.register_system(ingest_archive_members_handler, command_type=IngestArchiveCommand)
        world.register_system(clear_archive_components_handler, command_type=ClearArchiveComponentsCommand)
        world.register_system(check_archive_handler, command_type=CheckArchiveCommand)
        world.register_system(check_archive_password_handler, command_type=CheckArchivePasswordCommand)
        world.register_system(remove_archive_password_handler, command_type=RemoveArchivePasswordCommand)
        world.register_system(reissue_archive_member_events_handler, command_type=ReissueArchiveMemberEventsCommand)

        # New binding systems
        world.register_system(bind_split_archive_handler, command_type=BindSplitArchiveCommand)
        world.register_system(check_split_archive_binding_handler, command_type=CheckSplitArchiveBindingCommand)

        # Discovery system
        world.register_system(discover_archive_path_siblings_handler, command_type=DiscoverPathSiblingsCommand)

        # Register Asset Operations
        world.register_asset_operation(ingest_archive_operation)
        world.register_asset_operation(set_archive_password_operation)
        world.register_asset_operation(bind_split_archive_operation)
