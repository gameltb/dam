from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam_fs.commands import GetAssetStreamCommand
from .commands import SetArchivePasswordCommand
from .systems import get_archive_asset_stream_handler, set_archive_password_handler

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
