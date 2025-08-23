from __future__ import annotations

from typing import TYPE_CHECKING

from dam.core.plugin import Plugin

if TYPE_CHECKING:
    from dam.core.world import World


class SemanticPlugin(Plugin):
    """
    A plugin for handling semantic search.
    """

    def build(self, world: "World") -> None:
        """
        Builds the semantic plugin.
        """
        from . import systems
        world.register_system(systems.handle_semantic_search_query)
        world.register_system(systems.handle_audio_search_query)
        world.register_system(systems.generate_embeddings_system)


__all__ = ["SemanticPlugin"]
