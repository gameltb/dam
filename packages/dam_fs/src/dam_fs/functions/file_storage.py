"""Provides content-addressable storage (CAS) functions for managing files."""

import hashlib
import logging
from pathlib import Path

from dam.core.config import WorldConfig

logger = logging.getLogger(__name__)

MIN_HASH_LENGTH = 4


def _get_storage_path_for_world(file_hash: str, world_config: WorldConfig) -> Path:
    """
    Construct the full path for a given file hash.

    Constructs the full path using a nested directory structure,
    specific to the asset storage path defined in the world_config.
    Example: <world_asset_storage_path>/ab/cd/ef123456...
    """
    logger.info(
        "[_get_storage_path_for_world] World: %s, Base path: %s, Hash: %s",
        world_config.name,
        Path(world_config.ASSET_STORAGE_PATH),
        file_hash,
    )

    base_path = Path(world_config.ASSET_STORAGE_PATH)

    if not file_hash or len(file_hash) < MIN_HASH_LENGTH:
        raise ValueError(f"File hash must be at least {MIN_HASH_LENGTH} characters long for storage path generation.")

    sub_dir_1 = file_hash[:2]
    sub_dir_2 = file_hash[2:4]
    file_name = file_hash

    full_path = base_path / sub_dir_1 / sub_dir_2 / file_name
    logger.info("[_get_storage_path_for_world] Constructed full path: %s", full_path)
    return full_path


def store_file(
    file_content: bytes,
    world_config: WorldConfig,
    original_filename: str | None = None,
) -> tuple[str, str]:
    """
    Store the given file content using a content-addressable scheme (SHA256 hash).

    This function stores the content into the specified world's asset storage,
    using the provided WorldConfig.

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
    if not content_hash or len(content_hash) < MIN_HASH_LENGTH:
        raise ValueError(
            f"Content hash must be at least {MIN_HASH_LENGTH} characters long for storage path generation."
        )
    sub_dir_1 = content_hash[:2]
    sub_dir_2 = content_hash[2:4]
    file_name_in_cas = content_hash
    physical_storage_path_suffix = str(Path(sub_dir_1) / sub_dir_2 / file_name_in_cas)

    storage_path = _get_storage_path_for_world(content_hash, world_config)
    logger.info(
        "[store_file] World: %s, Original: %s, Hash: %s, Target storage_path: %s",
        world_config.name,
        original_filename,
        content_hash,
        storage_path,
    )
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    log_world_identifier = world_config.name

    if not storage_path.exists():
        with storage_path.open("wb") as f:
            f.write(file_content)
        logger.info(
            "Stored file %s to %s in world '%s'",
            original_filename or content_hash,
            storage_path,
            log_world_identifier,
        )
    else:
        logger.debug(
            "File %s (hash: %s) already exists at %s in world '%s'",
            original_filename or content_hash,
            content_hash,
            storage_path,
            log_world_identifier,
        )

    return content_hash, physical_storage_path_suffix


def has_file(file_identifier: str, world_config: WorldConfig) -> bool:
    """Check if a file with the given identifier (SHA256 hash) exists in storage."""
    return get_file_path(file_identifier, world_config) is not None


def get_file_path(file_identifier: str, world_config: WorldConfig) -> Path | None:
    """
    Return the absolute path to the file identified by file_identifier (SHA256 hash).

    The path is resolved from the specified world's asset storage, using the provided WorldConfig.

    Args:
        file_identifier: The SHA256 hash of the file.
        world_config: The configuration of the world to get the file from.

    Returns:
        The absolute Path object to the file if it exists, otherwise None.

    """
    if not file_identifier:
        return None

    try:
        storage_path = _get_storage_path_for_world(file_identifier, world_config)
        logger.info(
            "[get_file_path] World: %s, Identifier: %s, Checking storage_path: %s",
            world_config.name,
            file_identifier,
            storage_path,
        )
        if storage_path.exists() and storage_path.is_file():
            return storage_path.resolve()
        return None
    except ValueError:
        logger.warning("Invalid file identifier format: %s", file_identifier)
        return None


def delete_file(file_identifier: str, world_config: WorldConfig) -> bool:
    """
    Delete the file identified by file_identifier (SHA256 hash).

    The file is deleted from the specified world's asset storage, using the
    provided WorldConfig. This function also attempts to remove empty parent directories.

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
            actual_file_path.unlink()
            logger.info("Deleted file %s from world '%s'", actual_file_path, log_world_identifier)

            parent_dir = actual_file_path.parent
            try:
                if not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                    logger.info("Removed empty directory %s from world '%s'", parent_dir, log_world_identifier)

                    grandparent_dir = parent_dir.parent
                    if grandparent_dir != Path(world_config.ASSET_STORAGE_PATH) and not any(grandparent_dir.iterdir()):
                        grandparent_dir.rmdir()
                        logger.info(
                            "Removed empty directory %s from world '%s'",
                            grandparent_dir,
                            log_world_identifier,
                        )
            except OSError as e:
                logger.debug(
                    "Could not remove parent directory for %s in world '%s': %s",
                    actual_file_path,
                    log_world_identifier,
                    e,
                )
            return True
        except OSError as e:
            logger.exception(
                "Error deleting file %s from world '%s': %s",
                actual_file_path,
                log_world_identifier,
                e,
            )
            return False
    else:
        logger.warning(
            "File with identifier %s not found in world '%s' for deletion.",
            file_identifier,
            log_world_identifier,
        )
        return False
