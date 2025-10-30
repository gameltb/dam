"""Systems for data migration."""

import logging
from pathlib import Path

from dam.core.systems import system
from dam.core.world import World
from dam.functions.paths import get_or_create_path_tree_from_path
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from sqlalchemy import select

from ..commands import MigratePathsCommand

logger = logging.getLogger(__name__)


@system(on_command=MigratePathsCommand)
async def migrate_paths_handler(
    cmd: MigratePathsCommand,  # noqa: ARG001
    world: World,
) -> None:
    """Migrate existing file paths to the new path tree structure."""
    logger.info("Starting path migration.")

    # Migrate filesystem paths
    await _migrate_filesystem_paths(world)

    # Migrate archive paths
    await _migrate_archive_paths(world)

    logger.info("Path migration complete.")


async def _migrate_filesystem_paths(world: World) -> None:
    logger.info("Migrating filesystem paths.")
    async with world.transaction() as tx:
        stmt = select(FileLocationComponent)
        result = await tx.session.execute(stmt)
        file_locations = result.scalars().all()

        for flc in file_locations:
            if not flc.url:
                continue

            try:
                file_path = Path(flc.url)
            except Exception:
                continue

            tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, file_path, "filesystem")
            flc.tree_entity_id = tree_entity_id
            flc.node_id = node_id


async def _migrate_archive_paths(world: World) -> None:
    logger.info("Migrating archive paths.")
    async with world.transaction() as tx:
        stmt = select(ArchiveMemberComponent)
        result = await tx.session.execute(stmt)
        archive_members = result.scalars().all()

        for member in archive_members:
            tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, member.path_in_archive, "archive")
            member.tree_entity_id = tree_entity_id
            member.node_id = node_id
