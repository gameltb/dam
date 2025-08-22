"""
This service provides centralized functions for calculating various types of hashes
for files, including content hashes (e.g., MD5, SHA256) and perceptual hashes
for images (e.g., aHash, pHash, dHash).
"""
import asyncio
import hashlib
import logging
from pathlib import Path

# Conditional imports for optional image hashing feature
_imagehash_available = False
_PIL_available = False
try:
    import imagehash
    _imagehash_available = True
    from PIL import Image
    _PIL_available = True
except ImportError:
    # This warning can be made more prominent or logged if necessary
    logging.getLogger(__name__).warning(
        "Optional dependencies ImageHash and/or Pillow not found. Perceptual image hashing will be disabled."
    )


logger = logging.getLogger(__name__)


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
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_md5(filepath: Path) -> str:
    """
    Calculates the MD5 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the MD5 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    md5_hash = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        return md5_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def generate_perceptual_hashes(image_filepath: Path) -> dict[str, str]:
    """
    Generates various perceptual hashes for an image file if ImageHash and Pillow are installed.

    Args:
        image_filepath: Path to the image file.

    Returns:
        A dictionary with hash_type as key and hex hash string as value.
        Example: {"phash": "...", "ahash": "...", "dhash": "..."}
        Returns empty dict if dependencies are missing, or image cannot be processed.
    """
    hashes = {}
    if not _imagehash_available or not _PIL_available:
        # Dependencies were not available at import time of this module
        return hashes

    try:
        img = Image.open(image_filepath)

        # pHash (Perceptual Hash)
        try:
            hashes["phash"] = str(imagehash.phash(img))
        except Exception as e_phash:  # More specific exceptions could be caught
            logger.warning(f"Could not generate pHash for {image_filepath.name}: {e_phash}", exc_info=True)

        # aHash (Average Hash)
        try:
            hashes["ahash"] = str(imagehash.average_hash(img))
        except Exception as e_ahash:
            logger.warning(f"Could not generate aHash for {image_filepath.name}: {e_ahash}", exc_info=True)

        # dHash (Difference Hash)
        try:
            hashes["dhash"] = str(imagehash.dhash(img))
        except Exception as e_dhash:
            logger.warning(f"Could not generate dHash for {image_filepath.name}: {e_dhash}", exc_info=True)

    except FileNotFoundError:
        logger.warning(f"Image file not found at {image_filepath} for perceptual hashing.")
    except Exception as e_open:  # Catches PIL.UnidentifiedImageError etc.
        logger.warning(
            f"Could not open or process image {image_filepath.name} for perceptual hashing: {e_open}", exc_info=True
        )

    return hashes


def calculate_sha1(filepath: Path) -> str:
    """
    Calculates the SHA1 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the SHA1 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha1_hash = hashlib.sha1()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha1_hash.update(byte_block)
        return sha1_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_crc32(filepath: Path) -> int:
    """
    Calculates the CRC32 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The CRC32 hash as an integer.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    import zlib
    crc32_hash = 0
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                crc32_hash = zlib.crc32(byte_block, crc32_hash)
        return crc32_hash
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


# --- Async Wrappers ---

async def calculate_sha256_async(filepath: Path) -> str:
    """Asynchronously calculates the SHA256 hash of a file."""
    return await asyncio.to_thread(calculate_sha256, filepath)


async def calculate_md5_async(filepath: Path) -> str:
    """Asynchronously calculates the MD5 hash of a file."""
    return await asyncio.to_thread(calculate_md5, filepath)


async def calculate_sha1_async(filepath: Path) -> str:
    """Asynchronously calculates the SHA1 hash of a file."""
    return await asyncio.to_thread(calculate_sha1, filepath)


async def calculate_crc32_async(filepath: Path) -> int:
    """Asynchronously calculates the CRC32 hash of a file."""
    return await asyncio.to_thread(calculate_crc32, filepath)


async def generate_perceptual_hashes_async(image_filepath: Path) -> dict[str, str]:
    """Asynchronously generates various perceptual hashes for an image file."""
    return await asyncio.to_thread(generate_perceptual_hashes, image_filepath)
