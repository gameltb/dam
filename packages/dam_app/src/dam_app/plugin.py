from dam.core.plugin import Plugin
from dam.core.world import World

from .systems.auto_tagging_system import auto_tag_entity_command_handler
from .systems.ingestion_systems import (
    asset_dispatcher_system,
    ingest_asset_stream_command_handler,
)
from .systems.metadata_systems import extract_metadata_command_handler


class AppPlugin(Plugin):
    def build(self, world: "World") -> None:
        # Register Command Handlers
        world.register_system(ingest_asset_stream_command_handler)
        world.register_system(auto_tag_entity_command_handler)
        world.register_system(extract_metadata_command_handler)

        # Register Event Listeners
        world.register_system(asset_dispatcher_system)
