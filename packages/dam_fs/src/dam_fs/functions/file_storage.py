"""Provides content-addressable storage (CAS) functions for managing files."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_HASH_LENGTH = 4


def _get_storage_path(base_path: Path, file_hash: str) -> Path:
    """
    Construct the full path for a given file hash within a base storage path.

    Constructs the full path using a nested directory structure.
    Example: <base_path>/ab/cd/ef123456...
    """
    if not file_hash or len(file_hash) < MIN_HASH_LENGTH:
        raise ValueError(f"File hash must be at least {MIN_HASH_LENGTH} characters long for storage path generation.")

    sub_dir_1 = file_hash[:2]
    sub_dir_2 = file_hash[2:4]
    file_name = file_hash

    full_path = base_path / sub_dir_1 / sub_dir_2 / file_name
    return full_path


def store_file(
    file_content: bytes,
    storage_path: Path,
    original_filename: str | None = None,
) -> tuple[str, str]:
    """
    Store the given file content using a content-addressable scheme (SHA256 hash).

    Args:
        file_content: The binary content of the file to store.
        storage_path: The root path for the asset storage.
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

    target_path = _get_storage_path(storage_path, content_hash)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        with target_path.open("wb") as f:
            f.write(file_content)
        logger.info(
            "Stored file %s to %s",
            original_filename or content_hash,
            target_path,
        )
    else:
        logger.debug(
            "File %s (hash: %s) already exists at %s",
            original_filename or content_hash,
            content_hash,
            target_path,
        )

    return content_hash, physical_storage_path_suffix


def has_file(file_identifier: str, storage_path: Path) -> bool:
    """Check if a file with the given identifier (SHA256 hash) exists in storage."""
    return get_file_path(file_identifier, storage_path) is not None


def get_file_path(file_identifier: str, storage_path: Path) -> Path | None:
    """
    Return the absolute path to the file identified by file_identifier (SHA256 hash).

    Args:
        file_identifier: The SHA256 hash of the file.
        storage_path: The root path for the asset storage.

    Returns:
        The absolute Path object to the file if it exists, otherwise None.
    """
    if not file_identifier:
        return None

    try:
        target_path = _get_storage_path(storage_path, file_identifier)
        if target_path.exists() and target_path.is_file():
            return target_path.resolve()
        return None
    except ValueError:
        logger.warning("Invalid file identifier format: %s", file_identifier)
        return None


def delete_file(file_identifier: str, storage_path: Path) -> bool:
    """
    Delete the file identified by file_identifier (SHA256 hash).

    This function also attempts to remove empty parent directories.

    Args:
        file_identifier: The SHA256 hash of the file.
        storage_path: The root path for the asset storage.

    Returns:
        True if the file was deleted, False otherwise.
    """
    actual_file_path = get_file_path(file_identifier, storage_path)

    if actual_file_path and actual_file_path.exists():
        try:
            actual_file_path.unlink()
            logger.info("Deleted file %s", actual_file_path)

            parent_dir = actual_file_path.parent
            try:
                if not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                    logger.info("Removed empty directory %s", parent_dir)

                    grandparent_dir = parent_dir.parent
                    if grandparent_dir != storage_path and not any(grandparent_dir.iterdir()):
                        grandparent_dir.rmdir()
                        logger.info("Removed empty directory %s", grandparent_dir)
            except OSError as e:
                logger.debug(
                    "Could not remove parent directory for %s: %s",
                    actual_file_path,
                    e,
                )
            return True
        except OSError as e:
            logger.exception(
                "Error deleting file %s: %s",
                actual_file_path,
                e,
            )
            return False
    else:
        logger.warning(
            "File with identifier %s not found for deletion.",
            file_identifier,
        )
        return False