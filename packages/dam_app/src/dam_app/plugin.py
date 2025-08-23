from dam.core.plugin import Plugin
from dam.core.stages import SystemStage
from dam.core.world import World
from dam_fs.events import FileStored

from .events import AssetStreamIngestionRequested
from .systems.auto_tagging_system import auto_tag_entities_system
from .systems.ingestion_systems import (
    asset_dispatcher_system,
    ingestion_request_system,
)
from .systems.metadata_systems import extract_metadata_on_asset_ingested


class AppPlugin(Plugin):
    def build(self, world: "World") -> None:
        # Register old systems
        world.register_system(auto_tag_entities_system, stage=SystemStage.CONTENT_ANALYSIS)
        world.register_system(extract_metadata_on_asset_ingested, stage=SystemStage.METADATA_EXTRACTION)

        # Register new event-driven ingestion systems
        world.register_system(ingestion_request_system, event_type=AssetStreamIngestionRequested)
        world.register_system(asset_dispatcher_system, event_type=FileStored)
