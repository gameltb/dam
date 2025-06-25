import hashlib
import logging  # Import logging module
import os
from pathlib import Path
from typing import Optional

from dam.core.config import settings

logger = logging.getLogger(__name__)  # Initialize logger


def _get_storage_path(file_hash: str, base_path: Optional[Path] = None) -> Path:
    """
    Constructs the full path for a given file hash using a nested directory structure.
    Example: <base_path>/ab/cd/ef123456...
    """
    if base_path is None:
        base_path = Path(settings.ASSET_STORAGE_PATH)

    if not file_hash or len(file_hash) < 4:
        raise ValueError("File hash must be at least 4 characters long.")

    # Use the first 4 characters of the hash to create two levels of subdirectories
    # e.g., hash "abcdef123..." -> path "<base_path>/ab/cd/ef123..."
    # The actual file will be named by its full hash.
    sub_dir_1 = file_hash[:2]
    sub_dir_2 = file_hash[2:4]
    file_name = file_hash

    return base_path / sub_dir_1 / sub_dir_2 / file_name


def store_file(file_content: bytes, original_filename: Optional[str] = None) -> str:
    """
    Stores the given file content using a content-addressable scheme (SHA256 hash).
    The file is stored in a nested directory structure derived from its hash.

    Args:
        file_content: The binary content of the file to store.
        original_filename: The original name of the file (optional, not used for storage path).

    Returns:
        The SHA256 hash of the file content, which serves as its unique identifier.
    """
    file_hash = hashlib.sha256(file_content).hexdigest()
    storage_path = _get_storage_path(file_hash)

    # Create the directory structure if it doesn't exist
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file if it doesn't already exist (content-addressable storage)
    if not storage_path.exists():
        with open(storage_path, "wb") as f:
            f.write(file_content)

    return file_hash


def get_file_path(file_identifier: str) -> Optional[Path]:
    """
    Returns the absolute path to the file identified by file_identifier (SHA256 hash).

    Args:
        file_identifier: The SHA256 hash of the file.

    Returns:
        The absolute Path object to the file if it exists, otherwise None.
    """
    if not file_identifier:
        return None

    try:
        storage_path = _get_storage_path(file_identifier)
        if storage_path.exists() and storage_path.is_file():
            return storage_path.resolve()
        return None
    except ValueError:  # Raised by _get_storage_path for short identifiers
        return None


def delete_file(file_identifier: str) -> bool:
    """
    Deletes the file identified by file_identifier (SHA256 hash).
    Also attempts to remove empty parent directories.

    Args:
        file_identifier: The SHA256 hash of the file.

    Returns:
        True if the file was deleted, False otherwise.
    """
    file_path = get_file_path(file_identifier)
    if file_path and file_path.exists():
        try:
            os.remove(file_path)

            # Attempt to remove parent directories if they are empty
            parent_dir = file_path.parent
            try:
                os.rmdir(parent_dir)
                # Attempt to remove grandparent directory
                grandparent_dir = parent_dir.parent
                os.rmdir(grandparent_dir)
            except OSError:
                # Directory not empty or other error, which is fine.
                pass
            return True
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
            return False
    return False
