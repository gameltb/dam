from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

from .commands import ExtractArchiveCommand
from .systems import extract_archive_handler

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
        world.register_system(extract_archive_handler, command_type=ExtractArchiveCommand)
