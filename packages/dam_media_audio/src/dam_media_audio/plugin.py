from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import ExtractAudioMetadataCommand
from .systems.audio_systems import add_audio_components_system


class AudioPlugin(Plugin):
    def build(self, world: World) -> None:
        world.register_system(add_audio_components_system, command_type=ExtractAudioMetadataCommand)
