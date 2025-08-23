from dam.core.plugin import Plugin
from dam.core.world import World

from .resource import SireResource


class SirePlugin(Plugin):
    def build(self, world: "World") -> None:
        sire_resource = SireResource()
        world.add_resource(sire_resource)
