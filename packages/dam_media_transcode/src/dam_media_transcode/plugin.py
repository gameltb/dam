from dam.core.plugin import Plugin
from dam.core.world import World

from .systems.evaluation_systems import execute_evaluation_run


class TranscodePlugin(Plugin):
    def build(self, world: World) -> None:
        world.register_system(execute_evaluation_run)
