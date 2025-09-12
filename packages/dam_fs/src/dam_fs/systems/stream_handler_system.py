import asyncio
import logging
from typing import BinaryIO, Annotated, Optional

from dam.commands import GetAssetStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World

from dam_fs.utils.url_utils import get_local_path_for_url

from ..models.file_location_component import FileLocationComponent

logger = logging.getLogger(__name__)


@system(on_command=GetAssetStreamCommand)
async def get_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> Optional[BinaryIO]:
    """
    Handles getting a stream for a standalone asset. This is the default handler.
    It assumes that the archive handler has already run and returned None.
    """
    # Get all file locations for the entity
    all_locations = await transaction.get_components(cmd.entity_id, FileLocationComponent)
    if not all_locations:
        # No location found, another handler might be responsible.
        return None

    # Find a valid local path
    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            is_file = False
            if potential_path:
                is_file = await asyncio.to_thread(potential_path.is_file)
            if potential_path and is_file:
                return await asyncio.to_thread(open, potential_path, "rb")
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Could not resolve or find file for URL '{loc.url}' for entity {cmd.entity_id}: {e}")
            continue

    # No valid local file found.
    return None
