from dam.core.plugin import Plugin
from dam.core.world import World
from dam.core.stages import SystemStage

from .systems.auto_tagging_system import auto_tag_entities_system
from .systems.metadata_systems import extract_metadata_on_asset_ingested


class AppPlugin(Plugin):
    def build(self, world: "World") -> None:
        world.register_system(auto_tag_entities_system, stage=SystemStage.CONTENT_ANALYSIS)
        world.register_system(extract_metadata_on_asset_ingested, stage=SystemStage.METADATA_EXTRACTION)
