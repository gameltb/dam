import asyncio  # E402: Moved to top
import logging  # Import logging module
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)  # Initialize logger at module level


def get_file_properties(filepath: Path) -> Tuple[str, int]:
    """
    Retrieves basic file properties.

    Args:
        filepath: Path to the file.

    Returns:
        A tuple containing (original_filename, file_size_bytes).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    original_filename = filepath.name
    file_size_bytes = filepath.stat().st_size
    return original_filename, file_size_bytes


def get_mime_type(filepath: Path) -> str:
    """
    Detects the MIME type of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The detected MIME type string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    import mimetypes
    import subprocess

    mime_type = None
    try:
        result = subprocess.run(
            ["file", "-b", "--mime-type", str(filepath)],
            capture_output=True,
            text=True,
            check=True,
        )
        mime_type = result.stdout.strip()
        logger.debug(f"MIME type from 'file' command for {filepath.name}: {mime_type}")
    except FileNotFoundError:
        logger.warning("'file' command not found. Falling back to mimetypes module.")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"'file' command failed for {filepath.name}: {e}. Output: {e.stderr}. Falling back to mimetypes."
        )
    except Exception as e:
        logger.warning(
            f"An unexpected error occurred while using 'file' command for {filepath.name}: {e}. Falling back to mimetypes."
        )

    if not mime_type:
        mime_type_guess, _ = mimetypes.guess_type(filepath)
        if mime_type_guess:
            mime_type = mime_type_guess
            logger.debug(f"MIME type from mimetypes for {filepath.name}: {mime_type}")
        else:
            mime_type = "application/octet-stream"
            logger.warning(
                f"mimetypes could not guess MIME type for {filepath.name}. Defaulting to {mime_type}."
            )
    return mime_type


# --- Async Wrappers ---
# import asyncio # E402: Moved to top


async def read_file_async(filepath: Path) -> bytes:
    """Asynchronously reads the entire content of a file."""
    return await asyncio.to_thread(filepath.read_bytes)


async def get_file_properties_async(filepath: Path) -> Tuple[str, int]:
    """Asynchronously retrieves basic file properties."""
    return await asyncio.to_thread(get_file_properties, filepath)


async def get_mime_type_async(filepath: Path) -> str:
    """Asynchronously detects the MIME type of a file."""
    return await asyncio.to_thread(get_mime_type, filepath)


