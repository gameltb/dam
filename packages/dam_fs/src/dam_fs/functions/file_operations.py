import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from dam.core.config import WorldConfig
from dam.core.transaction import EcsTransaction
from dam.functions import ecs_functions
from dam.models.core.entity import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component

from dam_fs.utils.url_utils import get_local_path_for_url

from ..models.file_location_component import FileLocationComponent
from ..models.file_properties_component import FilePropertiesComponent
from ..resources.file_storage_resource import FileStorageResource

logger = logging.getLogger(__name__)


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
            logger.warning(f"mimetypes could not guess MIME type for {filepath.name}. Defaulting to {mime_type}.")
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


async def create_entity_with_file(transaction: EcsTransaction, world_config: WorldConfig, file_path: str) -> Entity:
    """
    Creates an entity for a given file path, with FileLocationComponent and FilePropertiesComponent.
    """
    p_file_path = Path(file_path)
    entity = await ecs_functions.create_entity(transaction.session)

    # Add FileLocationComponent
    # For now, we only support local files. The URL will be a file URI.
    file_url = p_file_path.as_uri()
    location_component = FileLocationComponent(
        url=file_url,
    )
    await ecs_functions.add_component_to_entity(transaction.session, entity.id, location_component)

    # Add FilePropertiesComponent
    original_filename, file_size_bytes = await get_file_properties_async(p_file_path)
    mime_type = await get_mime_type_async(p_file_path)
    properties_component = FilePropertiesComponent(
        original_filename=original_filename,
        size_bytes=file_size_bytes,
        mime_type=mime_type,
    )
    await ecs_functions.add_component_to_entity(transaction.session, entity.id, properties_component)

    return entity


async def get_file_path_by_id(transaction: EcsTransaction, file_id: int) -> Optional[Path]:
    """
    Resolves a file_id (which is the ID of a FilePropertiesComponent) to a local, accessible file path.
    """
    fpc = await transaction.get_component(file_id, FilePropertiesComponent)
    if not fpc:
        logger.warning(f"Could not find FilePropertiesComponent with ID {file_id}.")
        return None

    return await get_file_path_for_entity(transaction, fpc.entity_id)


async def get_file_path_for_entity(
    transaction: EcsTransaction, entity_id: int, variant_name: Optional[str] = "original"
) -> Optional[Path]:
    """
    Retrieves the full file path for a given entity.
    """
    logger.debug(f"Attempting to get file path for entity {entity_id}, variant {variant_name}")

    # 1. Try to get path from FileStorageResource using content hash
    sha256_comp = await transaction.get_component(entity_id, ContentHashSHA256Component)
    if sha256_comp:
        file_storage = transaction.world.get_resource(FileStorageResource)
        if file_storage:
            content_hash = sha256_comp.hash_value.hex()
            potential_path = file_storage.get_file_path(content_hash)
            if potential_path and await asyncio.to_thread(potential_path.is_file):
                logger.debug(f"Found file in FileStorageResource for entity {entity_id} at {potential_path}")
                return potential_path

    # 2. Fallback to FileLocationComponent
    all_locations = await transaction.get_components(entity_id, FileLocationComponent)
    if not all_locations:
        logger.warning(f"No FileLocationComponent found for entity {entity_id}.")
        return None

    # TODO: Add proper variant handling logic here
    target_loc = all_locations[0]

    try:
        potential_path = get_local_path_for_url(target_loc.url)
        if potential_path and await asyncio.to_thread(potential_path.is_file):
            logger.debug(f"Found file via FileLocationComponent for entity {entity_id} at {potential_path}")
            return potential_path
    except (ValueError, FileNotFoundError) as e:
        logger.debug(f"Could not resolve or find file for URL '{target_loc.url}' for entity {entity_id}: {e}")

    return None
