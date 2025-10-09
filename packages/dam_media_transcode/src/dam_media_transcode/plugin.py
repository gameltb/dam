"""Defines the plugin for transcoding-related functionality in the DAM system."""
from dam.core.plugin import Plugin
from dam.core.world import World

from .systems.evaluation_systems import execute_evaluation_run


class TranscodePlugin(Plugin):
    """A plugin to integrate transcoding-specific systems into the DAM."""

    def build(self, world: World) -> None:
        """Register transcoding-related systems with the world."""
        world.register_system(execute_evaluation_run)
