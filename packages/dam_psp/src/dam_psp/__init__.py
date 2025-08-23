from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

if TYPE_CHECKING:
    from dam.core.world import World


class PspPlugin(Plugin):
    """
    A plugin for handling PSP ISOs.
    """

    def build(self, world: "World") -> None:
        """
        Builds the PSP plugin.
        """
        # For now, this plugin doesn't register any systems.
        # The ingestion logic is called directly from the CLI.
        pass


__all__ = ["PspPlugin"]
