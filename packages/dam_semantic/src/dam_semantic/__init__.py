"""The `dam_semantic` package provides semantic search functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam_media_audio.commands import AudioSearchCommand

from . import systems
from .commands import SemanticSearchCommand
from .settings import SemanticSettingsComponent, SemanticSettingsModel

if TYPE_CHECKING:
    from dam.core.world import World


from dam_media_image.plugin import ImagePlugin

if TYPE_CHECKING:
    pass

sire_installed = False
try:
    __import__("dam_sire.resource")
    sire_installed = True
except ImportError:
    pass


class SemanticPlugin(Plugin):
    """A plugin for handling semantic search."""

    Settings = SemanticSettingsModel
    SettingsComponent = SemanticSettingsComponent

    def build(self, world: World) -> None:
        """
        Build the semantic plugin.

        Args:
            world: The world to build the plugin in.

        """
        # Add dependent plugins
        world.add_plugin(ImagePlugin())

        world.register_system(
            systems.handle_semantic_search_command,
            command_type=SemanticSearchCommand,
        )
        world.register_system(systems.handle_audio_search_command, command_type=AudioSearchCommand)
        world.register_system(systems.generate_embeddings_system)

        # No need to register the model type with SireResource, as it handles
        # torch.nn.Module subclasses by default, and transformers models are
        # torch.nn.Module subclasses.


__all__ = ["SemanticPlugin"]
