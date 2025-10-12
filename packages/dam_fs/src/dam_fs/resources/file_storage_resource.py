"""Defines the FileStorageResource for world-specific file operations."""

from pathlib import Path

from dam.core.world import World

from dam_fs.settings import FsSettingsComponent

from ..functions import file_storage  # Import the module with functions


class FileStorageResource:
    """
    A resource class that provides file storage operations, configured per world.

    It wraps the functional implementation in `dam_fs.functions.file_storage`.
    """

    def _get_storage_path(self, world: World) -> Path:
        """Get the configured asset storage path from the world's resources."""
        settings = world.get_resource(FsSettingsComponent)
        return Path(settings.ASSET_STORAGE_PATH)

    def store_file(self, world: World, file_content: bytes, original_filename: str | None = None) -> tuple[str, str]:
        """
        Store file content in this world's configured storage.

        Delegates to `file_storage.store_file`.
        """
        storage_path = self._get_storage_path(world)
        return file_storage.store_file(
            file_content=file_content, storage_path=storage_path, original_filename=original_filename
        )

    def get_file_path(self, world: World, file_identifier: str) -> Path | None:
        """
        Get the path to a file in this world's configured storage.

        Delegates to `file_storage.get_file_path`.
        """
        storage_path = self._get_storage_path(world)
        return file_storage.get_file_path(file_identifier=file_identifier, storage_path=storage_path)

    def has_file(self, world: World, file_identifier: str) -> bool:
        """Check if a file exists in this world's configured storage."""
        storage_path = self._get_storage_path(world)
        return file_storage.has_file(file_identifier=file_identifier, storage_path=storage_path)

    def delete_file(self, world: World, file_identifier: str) -> bool:
        """
        Delete a file from this world's configured storage.

        Delegates to `file_storage.delete_file`.
        """
        storage_path = self._get_storage_path(world)
        return file_storage.delete_file(file_identifier=file_identifier, storage_path=storage_path)
