import asyncio
import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple

from dam.core.config import WorldConfig
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.core.entity import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent

from dam_fs.utils.url_utils import get_local_path_for_url

from ..models.file_location_component import FileLocationComponent
from ..models.filename_component import FilenameComponent
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
    Creates an entity for a given file path, with file-related components.
    Note: This is a simplified helper and does not perform content-based deduplication.
    """
    p_file_path = Path(file_path)
    entity = await ecs_functions.create_entity(transaction.session)

    file_stat = p_file_path.stat()
    mod_time = datetime.datetime.fromtimestamp(file_stat.st_mtime, tz=datetime.timezone.utc).replace(microsecond=0)

    # Add FileLocationComponent
    file_url = p_file_path.as_uri()
    location_component = FileLocationComponent(
        url=file_url,
        last_modified_at=mod_time,
    )
    await ecs_functions.add_component_to_entity(transaction.session, entity.id, location_component)

    # Add FilenameComponent
    filename_component = FilenameComponent(
        filename=p_file_path.name,
        first_seen_at=mod_time,
    )
    await ecs_functions.add_component_to_entity(transaction.session, entity.id, filename_component)

    # Add ContentLengthComponent
    content_length_component = ContentLengthComponent(
        file_size_bytes=file_stat.st_size,
    )
    await ecs_functions.add_component_to_entity(transaction.session, entity.id, content_length_component)

    return entity


async def get_file_path_by_id(world: World, transaction: EcsTransaction, file_id: int) -> Optional[Path]:
    """
    Resolves a file_id (which is the ID of a FilenameComponent) to a local, accessible file path.
    TODO: This function is fragile as it assumes file_id is a component ID. Refactor to use entity_id.
    """
    fnc = await transaction.get_component(file_id, FilenameComponent)
    if not fnc:
        logger.warning(f"Could not find FilenameComponent with ID {file_id}.")
        return None

    return await get_file_path_for_entity(world, transaction, fnc.entity_id)


async def get_file_path_for_entity(
    world: World, transaction: EcsTransaction, entity_id: int, variant_name: Optional[str] = "original"
) -> Optional[Path]:
    """
    Retrieves the full file path for a given entity.
    """
    logger.debug(f"Attempting to get file path for entity {entity_id}, variant {variant_name}")

    # 1. Try to get path from FileStorageResource using content hash
    sha256_comp = await transaction.get_component(entity_id, ContentHashSHA256Component)
    if sha256_comp:
        file_storage = world.get_resource(FileStorageResource)
        if file_storage:
            content_hash = sha256_comp.hash_value.hex()
            potential_path = file_storage.get_file_path(content_hash)
            is_file = False
            if potential_path:
                is_file = await asyncio.to_thread(potential_path.is_file)
            if potential_path and is_file:
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
        is_file = False
        if potential_path:
            is_file = await asyncio.to_thread(potential_path.is_file)
        if potential_path and is_file:
            logger.debug(f"Found file via FileLocationComponent for entity {entity_id} at {potential_path}")
            return potential_path
    except (ValueError, FileNotFoundError) as e:
        logger.debug(f"Could not resolve or find file for URL '{target_loc.url}' for entity {entity_id}: {e}")

    return None
