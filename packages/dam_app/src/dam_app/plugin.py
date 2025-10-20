"""Defines the DAM application plugin and its lifecycle hooks."""

from dam.core.plugin import Plugin
from dam.core.world import World

from .operations import extract_exif_operation
from .settings import AppSettingsComponent, AppSettingsModel
from .systems.auto_tagging_system import auto_tag_entity_command_handler
from .systems.ingestion_systems import asset_dispatcher_system
from .systems.metadata_systems import (
    check_exif_metadata_handler,
    exiftool_instance,
    extract_metadata_command_handler,
    remove_exif_metadata_handler,
)


class AppPlugin(Plugin):
    """The main plugin for the DAM application."""

    Settings = AppSettingsModel
    SettingsComponent = AppSettingsComponent

    def build(self, world: "World") -> None:
        """Register all systems, commands, and event listeners for the app plugin."""
        # Register Command Handlers
        world.register_system(auto_tag_entity_command_handler)
        world.register_system(extract_metadata_command_handler)
        world.register_system(check_exif_metadata_handler)
        world.register_system(remove_exif_metadata_handler)

        # Register Event Listeners
        world.register_system(asset_dispatcher_system)

        # Register Asset Operations
        world.register_asset_operation(extract_exif_operation)

    async def on_stop(self, _world: "World"):
        """Stop the persistent exiftool process when the world shuts down."""
        await exiftool_instance.stop()
