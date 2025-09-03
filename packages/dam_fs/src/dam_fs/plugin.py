from dam.core.plugin import Plugin
from dam.core.world import World

from .commands import (
    FindEntityByHashCommand,
    GetAssetStreamCommand,
    IngestFileCommand,
    IngestReferenceCommand,
)
from .systems import get_asset_stream_handler
from .systems.asset_lifecycle_systems import (
    handle_find_entity_by_hash_command,
    handle_ingest_file_command,
    handle_ingest_reference_command,
)


class FsPlugin(Plugin):
    def build(self, world: "World") -> None:
        world.register_system(
            handle_ingest_file_command,
            command_type=IngestFileCommand,
        )
        world.register_system(
            handle_ingest_reference_command,
            command_type=IngestReferenceCommand,
        )
        world.register_system(handle_find_entity_by_hash_command, command_type=FindEntityByHashCommand)
        world.register_system(get_asset_stream_handler, command_type=GetAssetStreamCommand)
