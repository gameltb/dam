"""Defines the DAM application plugin and its lifecycle hooks."""

from dam.core.plugin import Plugin
from dam.core.world import World
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
from dam.traits.asset_operation import AssetOperationTrait
from dam.traits.traits import TraitImplementation

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
        extract_exif_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: extract_metadata_command_handler,
                AssetOperationTrait.Check: check_exif_metadata_handler,
                AssetOperationTrait.Remove: remove_exif_metadata_handler,
            },
            name="extract-exif-metadata",
            description="Extracts EXIF metadata from image files.",
        )
        world.trait_manager.register(extract_exif_implementation, ContentMimeTypeComponent)

    async def on_stop(self, _world: "World"):
        """Stop the persistent exiftool process when the world shuts down."""
        await exiftool_instance.stop()
