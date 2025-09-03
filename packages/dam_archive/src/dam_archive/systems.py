import asyncio
import logging
from typing import Annotated

from dam.core.systems import handles_command
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.utils.url_utils import get_local_path_for_url
from dam_app.commands import GetAssetStreamCommand
from dam_app.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent

from .main import open_archive

logger = logging.getLogger(__name__)


@handles_command(GetAssetStreamCommand)
async def get_archive_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
):
    """
    Handles getting a stream for an asset that is part of an archive.
    """
    world_config = world.world_config
    archive_member_component = await transaction.get_component(cmd.entity_id, ArchiveMemberComponent)

    if not archive_member_component:
        return None  # This handler only deals with assets in archives

    target_entity_id = archive_member_component.archive_entity_id
    path_in_archive = archive_member_component.path_in_archive

    all_locations = await transaction.get_components(target_entity_id, FileLocationComponent)
    if not all_locations:
        logger.warning(f"No FileLocationComponent found for archive entity {target_entity_id}.")
        return None

    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url, world_config)
            if potential_path and await asyncio.to_thread(potential_path.is_file):
                archive = open_archive(str(potential_path))
                if archive:
                    return archive.open_file(path_in_archive)
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Could not resolve or find file for URL '{loc.url}' for entity {target_entity_id}: {e}")
            continue

    return None  # No valid local file found for the archive
