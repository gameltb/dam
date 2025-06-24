import hashlib
import os
from pathlib import Path
from typing import Tuple

def calculate_sha256(filepath: Path) -> str:
    """
    Calculates the SHA256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the SHA256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}")


def get_file_properties(filepath: Path) -> Tuple[str, int, str]:
    """
    Retrieves basic file properties.

    Args:
        filepath: Path to the file.

    Returns:
        A tuple containing (original_filename, file_size_bytes, mime_type).
        MIME type detection is basic and relies on file extension.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    original_filename = filepath.name
    file_size_bytes = filepath.stat().st_size

    # Basic MIME type detection using mimetypes module
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        # Fallback or default if mimetypes can't guess
        # For example, use a generic type or raise an error
        mime_type = "application/octet-stream" # Generic binary type

    return original_filename, file_size_bytes, mime_type

# Placeholder for simulated file storage
def store_file_locally(source_filepath: Path, storage_base_path: Path, content_hash: str) -> Path:
    """
    Simulates storing a file locally using its content hash for organization.
    In a real system, this would involve more robust storage, error handling,
    and potentially structuring files by hash (e.g., aa/bb/aabbcc...).

    For now, it just prints a message.

    Args:
        source_filepath: The path to the original file.
        storage_base_path: The base directory where files are "stored".
        content_hash: The SHA256 hash of the file.

    Returns:
        The "path" where the file is supposedly stored.
    """
    # Example: storage_path = storage_base_path / content_hash[:2] / content_hash[2:4] / content_hash
    # For now, just simulate.
    simulated_path = storage_base_path / content_hash[:2] / content_hash

    # Ensure the simulated directory structure exists (optional for pure simulation)
    # (simulated_path.parent).mkdir(parents=True, exist_ok=True)
    # import shutil
    # shutil.copy2(source_filepath, simulated_path)

    print(f"[SIMULATED STORAGE] File '{source_filepath.name}' with hash '{content_hash}' would be stored at '{simulated_path}'.")

    # Return a relative path from the storage_base_path for the FileLocationComponent
    return simulated_path.relative_to(storage_base_path)


# Update services/__init__.py
# No, this file itself is fine. The __init__.py for services will be updated later if needed.
