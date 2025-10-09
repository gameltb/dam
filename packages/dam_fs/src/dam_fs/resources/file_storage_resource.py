"""Defines the FileStorageResource for world-specific file operations."""

from pathlib import Path

from dam.core.config import WorldConfig

from ..functions import file_storage  # Import the module with functions


class FileStorageResource:
    """
    A resource class that provides file storage operations, configured per world.

    It wraps the functional implementation in `dam_fs.functions.file_storage`.
    """

    def __init__(self, world_config: WorldConfig):
        """
        Initialize the FileStorageResource.

        Args:
            world_config: The configuration of the world.

        """
        self.world_config = world_config

    def get_world_asset_storage_path(self) -> Path:
        """
        Return the root asset storage path for the current world.

        It looks for a 'storage_path' under the [plugin_settings.dam-fs]
        section of the world's config. Falls back to './default_dam_storage'
        if not specified.
        """
        # Safely access nested dictionaries
        fs_settings = self.world_config.plugin_settings.get("dam-fs", {})
        storage_path = fs_settings.get("storage_path", "./default_dam_storage")
        return Path(storage_path)

    def store_file(self, file_content: bytes, original_filename: str | None = None) -> tuple[str, str]:
        """
        Store file content in this world's configured storage.

        Delegates to `file_storage.store_file`.
        """
        storage_path = self.get_world_asset_storage_path()
        return file_storage.store_file(
            file_content=file_content, storage_path=storage_path, original_filename=original_filename
        )

    def get_file_path(self, file_identifier: str) -> Path | None:
        """
        Get the path to a file in this world's configured storage.

        Delegates to `file_storage.get_file_path`.
        """
        storage_path = self.get_world_asset_storage_path()
        return file_storage.get_file_path(file_identifier=file_identifier, storage_path=storage_path)

    def has_file(self, file_identifier: str) -> bool:
        """Check if a file exists in this world's configured storage."""
        storage_path = self.get_world_asset_storage_path()
        return file_storage.has_file(file_identifier=file_identifier, storage_path=storage_path)

    def delete_file(self, file_identifier: str) -> bool:
        """
        Delete a file from this world's configured storage.

        Delegates to `file_storage.delete_file`.
        """
        storage_path = self.get_world_asset_storage_path()
        return file_storage.delete_file(file_identifier=file_identifier, storage_path=storage_path)
