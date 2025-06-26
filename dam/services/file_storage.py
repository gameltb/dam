import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from dam.core.config import WorldConfig

logger = logging.getLogger(__name__)


def _get_storage_path_for_world(file_hash: str, world_config: WorldConfig) -> Path:
    """
    Constructs the full path for a given file hash using a nested directory structure,
    specific to the asset storage path defined in the world_config.
    Example: <world_asset_storage_path>/ab/cd/ef123456...
    """
    base_path = Path(world_config.ASSET_STORAGE_PATH)

    if not file_hash or len(file_hash) < 4:
        raise ValueError("File hash must be at least 4 characters long for storage path generation.")

    sub_dir_1 = file_hash[:2]
    sub_dir_2 = file_hash[2:4]
    file_name = file_hash

    return base_path / sub_dir_1 / sub_dir_2 / file_name


def store_file(
    file_content: bytes,
    world_config: WorldConfig,
    original_filename: Optional[str] = None,
) -> tuple[str, str]:
    """
    Stores the given file content using a content-addressable scheme (SHA256 hash)
    into the specified world's asset storage, using the provided WorldConfig.

    Args:
        file_content: The binary content of the file to store.
        world_config: The configuration of the world to store the file in.
        original_filename: The original name of the file (optional, not used for storage path).

    Returns:
        A tuple containing:
            - The SHA256 hash of the file content (content_hash).
            - The relative physical storage path suffix (e.g., "ab/cd/hashvalue").
    """
    content_hash = hashlib.sha256(file_content).hexdigest()

    # Construct the relative path suffix for CAS based on the hash
    # This logic is similar to _get_storage_path_for_world but only for the suffix part
    if not content_hash or len(content_hash) < 4:
        raise ValueError("Content hash must be at least 4 characters long for storage path generation.")
    sub_dir_1 = content_hash[:2]
    sub_dir_2 = content_hash[2:4]
    file_name_in_cas = content_hash
    # The physical_storage_path_suffix is the path relative to the CAS root for this file
    physical_storage_path_suffix = str(Path(sub_dir_1) / sub_dir_2 / file_name_in_cas)

    # Get the full absolute storage path using the existing helper
    storage_path = _get_storage_path_for_world(content_hash, world_config)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    log_world_identifier = world_config.name

    if not storage_path.exists():
        with open(storage_path, "wb") as f:
            f.write(file_content)
        logger.info(
            f"Stored file {original_filename or content_hash} to {storage_path} in world '{log_world_identifier}'"
        )
    else:
        logger.debug(
            f"File {original_filename or content_hash} (hash: {content_hash}) already exists at {storage_path} in world '{log_world_identifier}'"
        )

    return content_hash, physical_storage_path_suffix


def get_file_path(file_identifier: str, world_config: WorldConfig) -> Optional[Path]:  # Changed from world_name
    """
    Returns the absolute path to the file identified by file_identifier (SHA256 hash)
    from the specified world's asset storage, using the provided WorldConfig.

    Args:
        file_identifier: The SHA256 hash of the file.
        world_config: The configuration of the world to get the file from.

    Returns:
        The absolute Path object to the file if it exists, otherwise None.
    """
    if not file_identifier:
        return None

    # world_config is now passed directly
    try:
        storage_path = _get_storage_path_for_world(file_identifier, world_config)
        if storage_path.exists() and storage_path.is_file():
            return storage_path.resolve()
        return None
    except ValueError:  # Raised by _get_storage_path_for_world for short identifiers
        logger.warning(f"Invalid file identifier format: {file_identifier}")
        return None


def delete_file(file_identifier: str, world_config: WorldConfig) -> bool:  # Changed from world_name
    """
    Deletes the file identified by file_identifier (SHA256 hash) from the
    specified world's asset storage, using the provided WorldConfig.
    Also attempts to remove empty parent directories.

    Args:
        file_identifier: The SHA256 hash of the file.
        world_config: The configuration of the world to delete the file from.

    Returns:
        True if the file was deleted, False otherwise.
    """
    actual_file_path = get_file_path(file_identifier, world_config)
    log_world_identifier = world_config.name

    if actual_file_path and actual_file_path.exists():
        try:
            os.remove(actual_file_path)
            logger.info(f"Deleted file {actual_file_path} from world '{log_world_identifier}'")

            # Attempt to remove empty parent directories
            parent_dir = actual_file_path.parent
            try:
                if not any(parent_dir.iterdir()):  # Check if directory is empty
                    os.rmdir(parent_dir)
                    logger.info(f"Removed empty directory {parent_dir} from world '{log_world_identifier}'")
                    # Try to remove grandparent directory as well if it's now empty
                    grandparent_dir = parent_dir.parent
                    # Ensure grandparent_dir is still within the world's asset storage path
                    # to avoid accidentally trying to remove directories outside of it.
                    # This check is a bit simplistic; Path.is_relative_to (Python 3.9+) or
                    # checking common prefix would be more robust.
                    # For now, assume the structure is <base_path>/xx/yy/hash
                    if grandparent_dir != Path(world_config.ASSET_STORAGE_PATH) and \
                       not any(grandparent_dir.iterdir()):
                        os.rmdir(grandparent_dir)
                        logger.info(
                            f"Removed empty directory {grandparent_dir} from world '{log_world_identifier}'"
                        )
            except OSError as e:
                # This is not critical if directories cannot be removed (e.g., not empty, permissions)
                logger.debug(
                    f"Could not remove parent directory for {actual_file_path} in world '{log_world_identifier}': {e}"
                )
            return True
        except OSError as e:
            logger.error(
                f"Error deleting file {actual_file_path} from world '{log_world_identifier}': {e}",
                exc_info=True,
            )
            return False
    else:
        logger.warning(
            f"File with identifier {file_identifier} not found in world '{log_world_identifier}' for deletion."
        )
        return False
