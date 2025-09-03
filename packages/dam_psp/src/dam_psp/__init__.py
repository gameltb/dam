from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam_app.events import AssetsReadyForMetadataExtraction

if TYPE_CHECKING:
    from dam.core.world import World


from . import psp_iso_functions
from .systems import psp_iso_metadata_extraction_system


class PspPlugin(Plugin):
    """
    A plugin for handling PSP ISOs.
    """

    def build(self, world: "World") -> None:
        """
        Builds the PSP plugin.
        """
        world.register_system(psp_iso_metadata_extraction_system, event_type=AssetsReadyForMetadataExtraction)


__all__ = ["PspPlugin", "psp_iso_functions"]
