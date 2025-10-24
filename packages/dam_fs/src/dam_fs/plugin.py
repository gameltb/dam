"""Plugin definition for the `dam_fs` package."""

from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    SetMimeTypeFromBufferCommand,
)
from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.plugin import Plugin
from dam.core.world import World
from dam.traits import TraitImplementation
from dam.traits.asset_content import AssetContentReadable

from .commands import (
    AddFilePropertiesCommand,
    FindEntityByFilePropertiesCommand,
    FindEntityByHashCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from .models.file_location_component import FileLocationComponent
from .resources.file_operations_resource import FileOperationsResource
from .resources.file_storage_resource import FileStorageResource
from .settings import FsSettingsComponent, FsSettingsModel
from .systems.asset_lifecycle_systems import (
    add_file_properties_handler,
    find_entity_by_file_properties_handler,
    get_fs_asset_filenames_handler,
    handle_find_entity_by_hash_command,
    register_local_file_handler,
    store_assets_handler,
)
from .systems.discovery_system import discover_fs_path_siblings_handler
from .systems.mime_type_system import (
    auto_set_mime_type_from_filename_system,
    set_mime_type_from_buffer_system,
)
from .systems.stream_handler_system import get_asset_stream_handler, get_size_from_file, get_stream_from_file


class FsPlugin(Plugin):
    """A plugin that provides filesystem-related functionalities."""

    Settings = FsSettingsModel
    SettingsComponent = FsSettingsComponent

    def build(self, world: "World") -> None:
        """
        Build the plugin by adding resources and systems to the world.

        Args:
            world: The world to build the plugin in.

        """
        # Add FileStorageResource
        settings = world.get_resource(FsSettingsComponent)
        file_storage_svc = FileStorageResource(settings)
        world.add_resource(file_storage_svc, FileStorageResource)
        world.logger.debug("Added FileStorageResource resource for World '%s'.", world.name)

        # Add FileOperationsResource
        world.add_resource(FileOperationsResource())
        world.logger.debug("Added FileOperationsResource for World '%s'.", world.name)

        world.register_system(
            add_file_properties_handler,
            command_type=AddFilePropertiesCommand,
        )
        world.register_system(handle_find_entity_by_hash_command, command_type=FindEntityByHashCommand)
        world.register_system(get_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(get_fs_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
        world.register_system(
            find_entity_by_file_properties_handler,
            command_type=FindEntityByFilePropertiesCommand,
        )
        world.register_system(
            register_local_file_handler,
            command_type=RegisterLocalFileCommand,
        )
        world.register_system(
            store_assets_handler,
            command_type=StoreAssetsCommand,
        )
        world.register_system(
            auto_set_mime_type_from_filename_system,
            command_type=AutoSetMimeTypeCommand,
        )
        world.register_system(
            set_mime_type_from_buffer_system,
            command_type=SetMimeTypeFromBufferCommand,
        )
        world.register_system(
            discover_fs_path_siblings_handler,
            command_type=DiscoverPathSiblingsCommand,
        )

        readable_impl = TraitImplementation(
            trait=AssetContentReadable,
            handlers={
                AssetContentReadable.GetStream: get_stream_from_file,
                AssetContentReadable.GetSize: get_size_from_file,
            },
            name="asset.content.readable",
            description="Provides a way to read the raw content of an asset.",
        )
        world.trait_manager.register(readable_impl, component_type=FileLocationComponent)
