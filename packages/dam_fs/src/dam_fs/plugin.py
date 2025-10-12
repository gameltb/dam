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

from .commands import (
    AddFilePropertiesCommand,
    FindEntityByFilePropertiesCommand,
    FindEntityByHashCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
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
from .systems.stream_handler_system import get_asset_stream_handler


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
        file_storage_svc = FileStorageResource()
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
