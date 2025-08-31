from pathlib import Path
from typing import Optional

from dam.core.config import WorldConfig

from dam_fs.functions import file_storage  # Import the module with functions


class FileStorageResource:
    """
    A resource class that provides file storage operations, configured per world.
    It wraps the functional implementation in `dam_fs.functions.file_storage`.
    """

    def __init__(self, world_config: WorldConfig):
        self.world_config = world_config

    def store_file(self, file_content: bytes, original_filename: Optional[str] = None) -> tuple[str, str]:
        """
        Stores file content in this world's configured storage.
        Delegates to `file_storage.store_file`.
        """
        return file_storage.store_file(
            file_content=file_content, world_config=self.world_config, original_filename=original_filename
        )

    def get_file_path(self, file_identifier: str) -> Optional[Path]:
        """
        Gets the path to a file in this world's configured storage.
        Delegates to `file_storage.get_file_path`.
        """
        return file_storage.get_file_path(file_identifier=file_identifier, world_config=self.world_config)

    def delete_file(self, file_identifier: str) -> bool:
        """
        Deletes a file from this world's configured storage.
        Delegates to `file_storage.delete_file`.
        """
        return file_storage.delete_file(file_identifier=file_identifier, world_config=self.world_config)

    def get_world_asset_storage_path(self) -> Path:
        """
        Returns the root asset storage path for the current world.
        """
        return Path(self.world_config.ASSET_STORAGE_PATH)
