from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import (
    AddFilePropertiesCommand,
    FindEntityByHashCommand,
    IngestFileCommand,
)
from .resources.file_operations_resource import FileOperationsResource
from .resources.file_storage_resource import FileStorageResource
from .systems.asset_lifecycle_systems import (
    add_file_properties_handler,
    get_fs_asset_filenames_handler,
    handle_find_entity_by_hash_command,
    handle_ingest_file_command,
)
from .systems.stream_handler_system import get_asset_stream_handler


class FsPlugin(Plugin):
    def build(self, world: "World") -> None:
        # Add FileStorageResource
        file_storage_svc = FileStorageResource(world_config=world.config)
        world.add_resource(file_storage_svc, FileStorageResource)
        world.logger.debug(f"Added FileStorageResource resource for World '{world.name}'.")

        # Add FileOperationsResource
        world.add_resource(FileOperationsResource())
        world.logger.debug(f"Added FileOperationsResource for World '{world.name}'.")

        world.register_system(
            handle_ingest_file_command,
            command_type=IngestFileCommand,
        )
        world.register_system(
            add_file_properties_handler,
            command_type=AddFilePropertiesCommand,
        )
        world.register_system(handle_find_entity_by_hash_command, command_type=FindEntityByHashCommand)
        world.register_system(get_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(get_fs_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
