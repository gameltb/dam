from dam.core.plugin import Plugin
from dam.core.world import World
from .events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
)

from .systems.asset_lifecycle_systems import (
    handle_asset_file_ingestion_request,
    handle_asset_reference_ingestion_request,
    handle_find_entity_by_hash_query,
)


class FsPlugin(Plugin):
    def build(self, world: "World") -> None:
        world.register_system(
            handle_asset_file_ingestion_request,
            event_type=AssetFileIngestionRequested,
        )
        world.register_system(
            handle_asset_reference_ingestion_request,
            event_type=AssetReferenceIngestionRequested,
        )
        world.register_system(
            handle_find_entity_by_hash_query, event_type=FindEntityByHashQuery
        )
