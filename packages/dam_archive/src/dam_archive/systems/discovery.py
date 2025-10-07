import logging
from pathlib import Path
from typing import List, Optional

from dam.commands.discovery_commands import DiscoverPathSiblingsCommand, PathSibling
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from sqlalchemy import select

from ..models import ArchiveMemberComponent

logger = logging.getLogger(__name__)


@system(on_command=DiscoverPathSiblingsCommand)
async def discover_archive_path_siblings_handler(
    cmd: DiscoverPathSiblingsCommand,
    transaction: WorldTransaction,
) -> Optional[List[PathSibling]]:
    """
    Handles discovering path-based sibling entities for an entity that is a member of an archive.
    """
    logger.debug("discover_archive_path_siblings_handler running for entity %s", cmd.entity_id)

    # 1. Get the ArchiveMemberComponent for the starting entity
    member_comp = await transaction.get_component(cmd.entity_id, ArchiveMemberComponent)
    if not member_comp or not member_comp.path_in_archive:
        logger.debug("Entity %s is not an archive member. Skipping archive discovery.", cmd.entity_id)
        return None

    # 2. Determine the parent archive and the directory within it
    parent_archive_id = member_comp.archive_entity_id
    internal_dir = Path(member_comp.path_in_archive).parent

    # 3. Find all members of the same parent archive first.
    stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == parent_archive_id)
    result = await transaction.session.execute(stmt)
    all_members = result.scalars().all()

    # 4. Filter them by directory and create PathSibling objects
    siblings = [
        PathSibling(entity_id=m.entity_id, path=m.path_in_archive)
        for m in all_members
        if m.path_in_archive and Path(m.path_in_archive).parent == internal_dir
    ]

    if siblings:
        logger.info(
            "Found %s archive siblings for entity %s in archive %s:%s.",
            len(siblings),
            cmd.entity_id,
            parent_archive_id,
            internal_dir,
        )
        return siblings

    return None
