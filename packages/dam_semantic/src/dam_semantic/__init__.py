"""The `dam_semantic` package provides semantic search functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin
from dam_media_audio.commands import AudioSearchCommand

from . import systems
from .commands import SemanticSearchCommand

if TYPE_CHECKING:
    from dam.core.world import World


from dam_media_image.plugin import ImagePlugin

if TYPE_CHECKING:
    from dam_sire.resource import SireResource
    from sentence_transformers import SentenceTransformer
    from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

try:
    from dam_sire.resource import SireResource
    from sentence_transformers import SentenceTransformer
    from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

    sire_installed = True
except ImportError:
    sire_installed = False


class SemanticPlugin(Plugin):
    """A plugin for handling semantic search."""

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

        # Register the SentenceTransformer model type with the SireResource
        if sire_installed:
            sire_resource = world.get_resource(SireResource)
            if sire_resource:
                sire_resource.register_model_type(SentenceTransformer, TorchModuleWrapper)


__all__ = ["SemanticPlugin"]
