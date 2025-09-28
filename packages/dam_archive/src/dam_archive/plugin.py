from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
)
from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import (
    CheckArchiveCommand,
    CheckArchivePasswordCommand,
    ClearArchiveComponentsCommand,
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    IngestArchiveCommand,
    RemoveArchivePasswordCommand,
    SetArchivePasswordCommand,
    UnbindSplitArchiveCommand,
)
from .operations import ingest_archive_operation, set_archive_password_operation
from .systems import (
    check_archive_handler,
    check_archive_password_handler,
    clear_archive_components_handler,
    create_master_archive_handler,
    discover_and_bind_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    ingest_archive_members_handler,
    remove_archive_password_handler,
    set_archive_password_handler,
    unbind_split_archive_handler,
)


class ArchivePlugin(Plugin):
    """
    A plugin for handling archives.
    """

    def build(self, world: World) -> None:
        """
        Builds the archive plugin.
        """
        # Command Handlers
        world.register_system(discover_and_bind_handler, command_type=DiscoverAndBindCommand)
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

        # Register Asset Operations
        world.register_asset_operation(ingest_archive_operation)
        world.register_asset_operation(set_archive_password_operation)
