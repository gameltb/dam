"""
This service provides functions for calculating perceptual hashes for images.
"""

import asyncio
import logging
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


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

    hashes: dict[str, str] = {}
    try:
        img = Image.open(image_filepath)

        # pHash (Perceptual Hash)
        try:
            hashes["phash"] = str(imagehash.phash(img))
        except Exception as e_phash:
            logger.warning(f"Could not generate pHash for {image_filepath.name}: {e_phash}", exc_info=True)

        # aHash (Average Hash)
        try:
            hashes["ahash"] = str(imagehash.average_hash(img))  # type: ignore
        except Exception as e_ahash:
            logger.warning(f"Could not generate aHash for {image_filepath.name}: {e_ahash}", exc_info=True)

        # dHash (Difference Hash)
        try:
            hashes["dhash"] = str(imagehash.dhash(img))
        except Exception as e_dhash:
            logger.warning(f"Could not generate dHash for {image_filepath.name}: {e_dhash}", exc_info=True)

    except FileNotFoundError:
        logger.warning(f"Image file not found at {image_filepath} for perceptual hashing.")
    except Exception as e_open:
        logger.warning(
            f"Could not open or process image {image_filepath.name} for perceptual hashing: {e_open}", exc_info=True
        )

    return hashes


async def generate_perceptual_hashes_async(image_filepath: Path) -> dict[str, str]:
    """Asynchronously generates various perceptual hashes for an image file."""
    return await asyncio.to_thread(generate_perceptual_hashes, image_filepath)
