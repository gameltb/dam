from dam.core.plugin import Plugin
from dam.core.world import World

from .operations import extract_exif_operation
from .systems.auto_tagging_system import auto_tag_entity_command_handler
from .systems.ingestion_systems import asset_dispatcher_system
from .systems.metadata_systems import (
    check_exif_metadata_handler,
    exiftool_instance,
    extract_metadata_command_handler,
    remove_exif_metadata_handler,
)


class AppPlugin(Plugin):
    def build(self, world: "World") -> None:
        # Register Command Handlers
        world.register_system(auto_tag_entity_command_handler)
        world.register_system(extract_metadata_command_handler)
        world.register_system(check_exif_metadata_handler)
        world.register_system(remove_exif_metadata_handler)

        # Register Event Listeners
        world.register_system(asset_dispatcher_system)

        # Register Asset Operations
        world.register_asset_operation(extract_exif_operation)

    async def on_stop(self, world: "World"):
        """Called when the world is shutting down."""
        await exiftool_instance.stop()
