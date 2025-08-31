from dam.core.plugin import Plugin
from dam.core.world import World

from .systems.audio_systems import add_audio_components_system

from dam.core.stages import SystemStage

import logging

logger = logging.getLogger(__name__)

class AudioPlugin(Plugin):
    def build(self, world: World) -> None:
        logger.info("Building AudioPlugin")
        world.register_system(add_audio_components_system, stage=SystemStage.METADATA_EXTRACTION)
