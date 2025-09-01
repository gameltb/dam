from dam.core.plugin import Plugin
from dam.core.world import World

from .systems.evaluation_systems import evaluation_system


class TranscodePlugin(Plugin):
    def build(self, world: World) -> None:
        world.register_system(evaluation_system)
