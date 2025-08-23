from dam.core.plugin import Plugin
from dam.core.world import World
from dam.core.events import WebAssetIngestionRequested

from .systems.web_asset_systems import handle_web_asset_ingestion_request


class SourcePlugin(Plugin):
    def build(self, world: "World") -> None:
        world.register_system(
            handle_web_asset_ingestion_request,
            event_type=WebAssetIngestionRequested,
        )
