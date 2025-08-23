from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

from .events import PspIsoAssetDetected

if TYPE_CHECKING:
    from dam.core.world import World


from .systems import process_psp_iso_system


class PspPlugin(Plugin):
    """
    A plugin for handling PSP ISOs.
    """

    def build(self, world: "World") -> None:
        """
        Builds the PSP plugin.
        """
        world.register_system(process_psp_iso_system, event_type=PspIsoAssetDetected)


__all__ = ["PspPlugin", "PspIsoAssetDetected"]
