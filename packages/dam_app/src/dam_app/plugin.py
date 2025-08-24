from dam.core.plugin import Plugin
from dam.core.stages import SystemStage
from dam.core.world import World
from dam_fs.events import FileStored

from .commands import AutoTagEntityCommand, IngestAssetStreamCommand
from .systems.auto_tagging_system import auto_tag_entity_command_handler
from .systems.ingestion_systems import (
    asset_dispatcher_system,
    ingest_asset_stream_command_handler,
)
from .systems.metadata_systems import extract_metadata_on_asset_ingested


class AppPlugin(Plugin):
    def build(self, world: "World") -> None:
        # Register Command Handlers
        world.register_system(
            ingest_asset_stream_command_handler, command_type=IngestAssetStreamCommand
        )
        world.register_system(
            auto_tag_entity_command_handler, command_type=AutoTagEntityCommand
        )

        # Register Event Listeners
        world.register_system(asset_dispatcher_system, event_type=FileStored)

        # Register Stage-based Systems
        world.register_system(extract_metadata_on_asset_ingested, stage=SystemStage.METADATA_EXTRACTION)
