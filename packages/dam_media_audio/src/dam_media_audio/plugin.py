"""Plugin definition for the `dam_media_audio` package."""

from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import ExtractAudioMetadataCommand
from .settings import AudioSettingsComponent, AudioSettingsModel
from .systems.audio_systems import add_audio_components_system


class AudioPlugin(Plugin):
    """A plugin that provides audio-related functionalities."""

    Settings = AudioSettingsModel
    SettingsComponent = AudioSettingsComponent

    def build(self, world: World) -> None:
        """
        Build the plugin by adding the audio systems to the world.

        Args:
            world: The world to build the plugin in.

        """
        world.register_system(add_audio_components_system, command_type=ExtractAudioMetadataCommand)
