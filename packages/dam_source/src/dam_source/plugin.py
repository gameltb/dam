from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import IngestWebAssetCommand
from .systems.web_asset_systems import handle_ingest_web_asset_command


class SourcePlugin(Plugin):
    def build(self, world: "World") -> None:
        world.register_system(
            handle_ingest_web_asset_command,
            command_type=IngestWebAssetCommand,
        )
