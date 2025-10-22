"""Defines the system for handling asset streams."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from dam.commands.asset_commands import GetAssetStreamCommand
from dam.core.database import DatabaseManager
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.types import FileStreamProvider, StreamProvider
from dam.core.world import World
from dam.traits.asset_content import AssetContentReadable

from ..models.file_location_component import FileLocationComponent
from ..utils.url_utils import get_local_path_for_url

logger = logging.getLogger(__name__)


@system(on_command=GetAssetStreamCommand)
async def get_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: WorldTransaction,
    _world: Annotated[World, "Resource"],
) -> StreamProvider | None:
    """
    Handle getting a stream for a standalone asset.

    This is the default handler. It assumes that the archive handler has
    already run and returned None.
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
                return FileStreamProvider(potential_path)
        except (ValueError, FileNotFoundError) as e:
            logger.debug(
                "Could not resolve or find file for URL '%s' for entity %s: %s",
                loc.url,
                cmd.entity_id,
                e,
            )
            continue

    # No valid local file found.
    return None


@system(on_command=AssetContentReadable.GetStream)
async def get_stream_from_file(
    cmd: AssetContentReadable.GetStream,
    world: World,
) -> StreamProvider | None:
    """Handle the GetStream command for entities with a FileLocationComponent."""
    db = world.get_resource(DatabaseManager)
    file_loc = await db.get_component(cmd.entity_id, FileLocationComponent)
    if not file_loc or not Path(file_loc.url).exists():
        return None

    return FileStreamProvider(Path(file_loc.url))


@system(on_command=AssetContentReadable.GetSize)
async def get_size_from_file(
    cmd: AssetContentReadable.GetSize,
    world: World,
) -> int:
    """Handle the GetSize command for entities with a FileLocationComponent."""
    db = world.get_resource(DatabaseManager)
    file_loc = await db.get_component(cmd.entity_id, FileLocationComponent)
    if not file_loc or not Path(file_loc.url).exists():
        return 0
    return Path(file_loc.url).stat().st_size
