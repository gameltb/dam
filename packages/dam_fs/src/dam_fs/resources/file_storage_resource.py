"""Defines the FileStorageResource for file operations."""

from pathlib import Path

from ..functions import file_storage
from ..settings import FsSettingsComponent


class FileStorageResource:
    """
    A resource class that provides file storage operations.

    It wraps the functional implementation in `dam_fs.functions.file_storage`
    and is configured with a specific storage path upon initialization.
    """

    def __init__(self, settings: FsSettingsComponent):
        """
        Initialize the FileStorageResource with world-specific settings.

        Args:
            settings: The FsSettingsComponent containing the asset_storage_path.

        """
        self._storage_path = Path(settings.asset_storage_path)

    def store_file(self, file_content: bytes, original_filename: str | None = None) -> tuple[str, str]:
        """
        Store file content in the configured storage.

        Delegates to `file_storage.store_file`.
        """
        return file_storage.store_file(
            file_content=file_content,
            storage_path=self._storage_path,
            original_filename=original_filename,
        )

    def get_file_path(self, file_identifier: str) -> Path | None:
        """
        Get the path to a file in the configured storage.

        Delegates to `file_storage.get_file_path`.
        """
        return file_storage.get_file_path(file_identifier=file_identifier, storage_path=self._storage_path)

    def has_file(self, file_identifier: str) -> bool:
        """Check if a file exists in the configured storage."""
        return file_storage.has_file(file_identifier=file_identifier, storage_path=self._storage_path)

    def delete_file(self, file_identifier: str) -> bool:
        """
        Delete a file from the configured storage.

        Delegates to `file_storage.delete_file`.
        """
        return file_storage.delete_file(file_identifier=file_identifier, storage_path=self._storage_path)
