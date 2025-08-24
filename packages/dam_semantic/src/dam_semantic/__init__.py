from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

from .commands import SemanticSearchCommand

if TYPE_CHECKING:
    from dam.core.world import World


from dam_media_image.plugin import ImagePlugin


class SemanticPlugin(Plugin):
    """
    A plugin for handling semantic search.
    """

    def build(self, world: "World") -> None:
        """
        Builds the semantic plugin.
        """
        # Add dependent plugins
        world.add_plugin(ImagePlugin())

        from dam_media_audio.commands import AudioSearchCommand

        from . import systems

        world.register_system(
            systems.handle_semantic_search_command,
            command_type=SemanticSearchCommand,
        )
        world.register_system(systems.handle_audio_search_command, command_type=AudioSearchCommand)
        world.register_system(systems.generate_embeddings_system)

        # Register the SentenceTransformer model type with the SireResource
        try:
            from dam_sire.resource import SireResource
            from sentence_transformers import SentenceTransformer
            from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

            sire_resource = world.resources.get(SireResource)
            if sire_resource:
                sire_resource.register_model_type(SentenceTransformer, TorchModuleWrapper)
        except ImportError:
            # dam_sire or sentence_transformers is not installed
            pass


__all__ = ["SemanticPlugin"]
