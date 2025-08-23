from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam.core.world_setup import register_core_systems

if TYPE_CHECKING:
    from dam.core.world import World


class DamPlugin(Plugin):
    """
    The core plugin for the DAM system.
    """

    def build(self, world: "World") -> None:
        """
        Builds the core DAM plugin.
        """
        register_core_systems(world)


__all__ = ["DamPlugin"]
