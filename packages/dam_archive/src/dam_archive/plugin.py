from typing import TYPE_CHECKING

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.plugin import Plugin

from .commands import ProcessArchiveCommand, SetArchivePasswordCommand
from .systems import (
    process_archive_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    set_archive_password_handler,
)

if TYPE_CHECKING:
    from dam.core.world import World


class ArchivePlugin(Plugin):
    """
    A plugin for handling archives.
    """

    def build(self, world: "World") -> None:
        """
        Builds the archive plugin.
        """
        world.register_system(get_archive_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(set_archive_password_handler, command_type=SetArchivePasswordCommand)
        world.register_system(get_archive_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
        world.register_system(process_archive_handler, command_type=ProcessArchiveCommand)
