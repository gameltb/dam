"""Plugin definition for the `dam_source` package."""

from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import IngestWebAssetCommand
from .systems.web_asset_systems import handle_ingest_web_asset_command


class SourcePlugin(Plugin):
    """A plugin that provides asset source tracking functionalities."""

    def build(self, world: "World") -> None:
        """
        Build the plugin by adding the web asset systems to the world.

        Args:
            world: The world to build the plugin in.

        """
        world.register_system(
            handle_ingest_web_asset_command,
            command_type=IngestWebAssetCommand,
        )
