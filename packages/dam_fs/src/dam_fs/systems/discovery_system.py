import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from dam.commands.discovery_commands import DiscoverPathSiblingsCommand, PathSibling
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from sqlalchemy import select

from ..models.file_location_component import FileLocationComponent

logger = logging.getLogger(__name__)


@system(on_command=DiscoverPathSiblingsCommand)
async def discover_fs_path_siblings_handler(
    cmd: DiscoverPathSiblingsCommand,
    transaction: WorldTransaction,
) -> list[PathSibling] | None:
    """Handles discovering path-based sibling entities for an entity located on the filesystem."""
    logger.debug(f"discover_fs_path_siblings_handler running for entity {cmd.entity_id}")

    # 1. Get the FileLocationComponent for the starting entity
    flc = await transaction.get_component(cmd.entity_id, FileLocationComponent)
    if not flc or not flc.url:
        logger.debug(f"Entity {cmd.entity_id} has no FileLocationComponent with a URL. Skipping fs discovery.")
        return None

    # 2. Determine the directory from the URL
    try:
        parsed_url = urlparse(flc.url)
        if parsed_url.scheme != "file":
            logger.debug(f"URL scheme for entity {cmd.entity_id} is not 'file'. Skipping fs discovery.")
            return None

        directory_path_str = os.path.dirname(unquote(parsed_url.path))
        # Ensure the directory path is absolute and normalized for the OS
        directory_path = Path(directory_path_str).resolve()
        directory_uri_prefix = directory_path.as_uri()

    except Exception as e:
        logger.error(f"Could not parse directory from URL '{flc.url}': {e}")
        return None

    # 3. Find all candidate entities in or below the directory
    stmt = select(FileLocationComponent).where(FileLocationComponent.url.like(f"{directory_uri_prefix}%"))
    result = await transaction.session.execute(stmt)
    all_candidates = result.scalars().all()

    # 4. Filter to include only direct children of the directory
    siblings: list[PathSibling] = []
    for component in all_candidates:
        if not component.url:
            continue
        try:
            parsed_url = urlparse(component.url)
            candidate_path = Path(unquote(parsed_url.path)).resolve()
            # Check if the candidate's parent directory is the same as the target directory
            if candidate_path.parent == directory_path:
                siblings.append(PathSibling(entity_id=component.entity_id, path=str(candidate_path)))
        except Exception:
            # Ignore candidates with malformed URLs or paths
            continue

    if siblings:
        logger.info(f"Found {len(siblings)} filesystem siblings for entity {cmd.entity_id} in '{directory_path}'.")
        return siblings

    return None
