from typing import TYPE_CHECKING

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.plugin import Plugin

from .commands import ExtractArchiveMembersCommand, SetArchivePasswordCommand, ClearArchiveComponentsCommand
from .systems import (
    extract_archive_members_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    set_archive_password_handler,
    clear_archive_components_handler,
)

from dam.core.world import World


class ArchivePlugin(Plugin):
    """
    A plugin for handling archives.
    """

    def build(self, world: World) -> None:
        """
        Builds the archive plugin.
        """
        world.register_system(get_archive_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(set_archive_password_handler, command_type=SetArchivePasswordCommand)
        world.register_system(get_archive_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
        world.register_system(extract_archive_members_handler, command_type=ExtractArchiveMembersCommand)
        world.register_system(clear_archive_components_handler, command_type=ClearArchiveComponentsCommand)
