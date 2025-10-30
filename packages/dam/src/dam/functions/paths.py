"""Functions for working with paths."""

from pathlib import Path

from sqlalchemy import and_, select

from dam.core.transaction import WorldTransaction
from dam.models.paths import PathNode, PathRoot


async def get_or_create_path_tree_from_path(
    transaction: WorldTransaction,
    path: str | Path,
    path_type: str,
) -> tuple[int, int]:
    """
    Create a path tree from a path string.

    Args:
        transaction: The world transaction.
        path: The path to create the tree from.
        path_type: The type of the path (e.g., 'filesystem', 'archive').

    Returns:
        A tuple of (tree_entity_id, node_id).

    """
    if isinstance(path, str):
        path = Path(path)

    path_root_comp = await transaction.get_unique_component(PathRoot, path_type=path_type)
    if not path_root_comp:
        root_entity = await transaction.create_entity()
        path_root_comp = PathRoot(path_type=path_type)
        await transaction.add_component_to_entity(root_entity.id, path_root_comp)
        tree_entity_id = root_entity.id
    else:
        tree_entity_id = path_root_comp.entity_id

    current_parent_id = None
    for segment in path.parent.parts:
        if not segment:
            continue
        stmt = select(PathNode).where(
            and_(
                PathNode.entity_id == tree_entity_id,
                PathNode.parent_id == current_parent_id,
                PathNode.segment == segment,
            )
        )
        result = await transaction.session.execute(stmt)
        path_node = result.scalar_one_or_none()

        if not path_node:
            path_node = await transaction.add_component_to_entity(
                tree_entity_id, PathNode(parent_id=current_parent_id, segment=segment)
            )
            await transaction.flush()
        current_parent_id = path_node.id

    stmt = select(PathNode).where(
        and_(
            PathNode.entity_id == tree_entity_id,
            PathNode.parent_id == current_parent_id,
            PathNode.segment == path.name,
        )
    )
    result = await transaction.session.execute(stmt)
    file_node = result.scalar_one_or_none()
    if not file_node:
        file_node = await transaction.add_component_to_entity(
            tree_entity_id, PathNode(parent_id=current_parent_id, segment=path.name)
        )
        await transaction.flush()

    return tree_entity_id, file_node.id
