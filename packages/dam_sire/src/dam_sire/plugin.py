"""Plugin definition for the `dam_sire` package."""

from dam.core.plugin import Plugin
from dam.core.world import World

from .resource import SireResource
from .settings import SireSettingsComponent, SireSettingsModel


class SirePlugin(Plugin):
    """A plugin that provides Sire integration."""

    Settings = SireSettingsModel
    SettingsComponent = SireSettingsComponent

    def build(self, world: "World") -> None:
        """
        Build the plugin by adding the SireResource to the world.

        Args:
            world: The world to build the plugin in.

        """
        sire_resource = SireResource()
        world.add_resource(sire_resource)
