"""Defines the system for discovering filesystem path siblings."""

import logging
from pathlib import Path

from dam.commands.discovery_commands import DiscoverPathSiblingsCommand, PathSibling
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.models.paths import PathNode
from sqlalchemy import and_, select

from ..models.file_location_component import FileLocationComponent

logger = logging.getLogger(__name__)


@system(on_command=DiscoverPathSiblingsCommand)
async def discover_fs_path_siblings_handler(
    cmd: DiscoverPathSiblingsCommand,
    transaction: WorldTransaction,
) -> list[PathSibling] | None:
    """Discover path-based sibling entities for an entity located on the filesystem."""
    logger.debug("discover_fs_path_siblings_handler running for entity %s", cmd.entity_id)

    # 1. Get the FileLocationComponent for the starting entity
    flc = await transaction.get_component(cmd.entity_id, FileLocationComponent)
    if not flc or not flc.tree_entity_id or not flc.node_id:
        logger.debug("Entity %s has no FileLocationComponent with a path tree. Skipping fs discovery.", cmd.entity_id)
        return None

    # 2. Get the PathNode for the entity
    stmt = select(PathNode).where(and_(PathNode.entity_id == flc.tree_entity_id, PathNode.id == flc.node_id))
    result = await transaction.session.execute(stmt)
    file_node = result.scalar_one_or_none()
    if not file_node:
        logger.debug("PathNode with id %s not found for entity %s. Skipping fs discovery.", flc.node_id, cmd.entity_id)
        return None

    # 3. Find all nodes with the same parent
    stmt = select(PathNode).where(PathNode.parent_id == file_node.parent_id)
    result = await transaction.session.execute(stmt)
    sibling_nodes = result.scalars().all()

    # 4. Find the corresponding entities and construct the path
    siblings: list[PathSibling] = []
    for node in sibling_nodes:
        # Find the entity that references this node
        stmt = select(FileLocationComponent.entity_id).where(FileLocationComponent.node_id == node.id)
        result = await transaction.session.execute(stmt)
        entity_id = result.scalar_one_or_none()

        if entity_id:
            # Reconstruct the path
            path_segments: list[str] = []
            current_node = node
            while True:
                path_segments.insert(0, current_node.segment)
                if current_node.parent_id is None:
                    break

                stmt = select(PathNode).where(
                    and_(PathNode.entity_id == flc.tree_entity_id, PathNode.id == current_node.parent_id)
                )
                result = await transaction.session.execute(stmt)
                parent_node = result.scalar_one_or_none()

                if not parent_node:
                    path_segments = []
                    break
                current_node = parent_node

            if not path_segments:
                continue

            path = str(Path(*path_segments))
            siblings.append(PathSibling(entity_id=entity_id, path=path))

    if siblings:
        logger.info(
            "Found %s filesystem siblings for entity %s.",
            len(siblings),
            cmd.entity_id,
        )
        return siblings

    return None
