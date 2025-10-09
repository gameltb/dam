"""Defines the plugin for image-related functionality in the DAM system."""
from dam.core.plugin import Plugin
from dam.core.world import World

from .resources import ImageHashingServiceResource
from .systems.image_systems import (
    handle_find_similar_images_command,
    process_image_metadata_system,
)


class ImagePlugin(Plugin):
    """A plugin to integrate image-specific systems and resources into the DAM."""

    def build(self, world: World) -> None:
        """Register image-related resources and systems with the world."""
        world.add_resource(ImageHashingServiceResource())
        world.register_system(handle_find_similar_images_command)
        world.register_system(process_image_metadata_system)
