from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import FindSimilarImagesCommand
from .events import ImageAssetDetected
from .resources import ImageHashingServiceResource
from .systems.image_systems import (
    add_image_components_system,
    handle_find_similar_images_command,
    process_image_metadata_system,
)


class ImagePlugin(Plugin):
    def build(self, world: World) -> None:
        world.add_resource(ImageHashingServiceResource())
        world.register_system(add_image_components_system)
        world.register_system(handle_find_similar_images_command, command_type=FindSimilarImagesCommand)
        world.register_system(process_image_metadata_system, event_type=ImageAssetDetected)
