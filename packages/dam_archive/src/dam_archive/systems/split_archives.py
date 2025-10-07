import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, Optional

from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
from dam_fs.models import FilenameComponent
from sqlalchemy import select

from .. import split_detector
from ..commands.split_archives import (
    BindSplitArchiveCommand,
    CheckSplitArchiveBindingCommand,
    CreateMasterArchiveCommand,
    UnbindSplitArchiveCommand,
)
from ..models import (
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)

logger = logging.getLogger(__name__)


@system(on_command=CreateMasterArchiveCommand)
async def create_master_archive_handler(
    cmd: CreateMasterArchiveCommand,
    transaction: WorldTransaction,
):
    """
    Handles the manual creation of a master entity for a split archive.
    """
    logger.info("Manually creating master archive '%s' for %s parts.", cmd.name, len(cmd.part_entity_ids))

    # 1. Create master entity and its components
    master_entity = await transaction.create_entity()
    await transaction.add_component_to_entity(
        master_entity.id,
        FilenameComponent(filename=cmd.name, first_seen_at=datetime.now(timezone.utc)),
    )
    manifest = SplitArchiveManifestComponent()
    await transaction.add_component_to_entity(master_entity.id, manifest)

    # 2. Copy mime type from first part
    if cmd.part_entity_ids:
        first_part_id = cmd.part_entity_ids[0]
        content_mime_comp = await transaction.get_component(first_part_id, ContentMimeTypeComponent)
        if content_mime_comp:
            await transaction.add_component_to_entity(
                master_entity.id,
                ContentMimeTypeComponent(mime_type_concept_id=content_mime_comp.mime_type_concept_id),
            )

    # 3. Update all part components to link to the new master
    for part_id in cmd.part_entity_ids:
        fnc = await transaction.get_component(part_id, FilenameComponent)
        if not fnc or not fnc.filename:
            logger.warning("Skipping part entity %s as it has no filename component.", part_id)
            continue

        split_info = split_detector.detect(fnc.filename)
        if not split_info:
            logger.warning("Skipping part entity %s as it does not look like a split archive part.", part_id)
            continue

        part_info_comp = await transaction.get_component(part_id, SplitArchivePartInfoComponent)
        if part_info_comp:
            part_info_comp.master_entity_id = master_entity.id
        else:
            new_part_info = SplitArchivePartInfoComponent(
                part_num=split_info.part_num,
                master_entity_id=master_entity.id,
            )
            await transaction.add_component_to_entity(part_id, new_part_info)

    logger.info("Successfully created master entity %s for archive '%s'.", master_entity.id, cmd.name)


@system(on_command=BindSplitArchiveCommand)
async def bind_split_archive_handler(
    cmd: BindSplitArchiveCommand,
    world: Annotated[World, "Resource"],
    transaction: WorldTransaction,
):
    """
    Handles discovering and binding a split archive from a starting entity.
    This is the "consumer" of the generic sibling discovery.
    """
    logger.info("Attempting to bind split archive starting from entity %s", cmd.entity_id)

    # 1. Discover sibling entities
    discover_cmd = DiscoverPathSiblingsCommand(entity_id=cmd.entity_id)
    try:
        siblings = await world.dispatch_command(discover_cmd).get_first_non_none_value()
    except ValueError:
        siblings = None

    if not siblings:
        logger.info("No siblings found for entity %s. Cannot bind split archive.", cmd.entity_id)
        return

    # 2. Use the split_detector to find and validate a complete group from the siblings
    parts_by_basename: Dict[str, Dict[int, int]] = {}  # {base_name: {part_num: entity_id}}
    for sibling in siblings:
        split_info = split_detector.detect(Path(sibling.path).name)
        if split_info:
            parts_by_basename.setdefault(split_info.base_name, {})[split_info.part_num] = sibling.entity_id

    # 3. Process the groups to find a complete one
    for base_name, parts_by_num in parts_by_basename.items():
        if not parts_by_num:
            continue

        max_part_num = max(parts_by_num.keys())
        is_complete = all(i in range(1, max_part_num + 1) for i in parts_by_num)

        if is_complete:
            logger.info("Found complete split archive '%s' with %s parts.", base_name, max_part_num)

            # Check if a master archive already exists for this group
            master_name = f"{base_name} (Split Archive)"
            stmt = select(FilenameComponent).where(FilenameComponent.filename == master_name)
            result = await transaction.session.execute(stmt)
            if result.scalar_one_or_none():
                logger.warning("A master archive named '%s' already exists. Skipping creation.", master_name)
                continue

            # 4. Dispatch command to create the master entity
            part_entity_ids = [parts_by_num[i] for i in range(1, max_part_num + 1)]
            create_cmd = CreateMasterArchiveCommand(
                name=master_name,
                part_entity_ids=part_entity_ids,
            )
            async for _ in world.dispatch_command(create_cmd):
                pass

            # We only bind the first complete group we find
            return


@system(on_command=CheckSplitArchiveBindingCommand)
async def check_split_archive_binding_handler(
    cmd: CheckSplitArchiveBindingCommand,
    transaction: WorldTransaction,
) -> bool:
    """
    Checks if an entity is part of a fully bound split archive.
    """
    # Case 1: The entity is the master archive itself.
    manifest = await transaction.get_component(cmd.entity_id, SplitArchiveManifestComponent)
    if manifest:
        return True

    # Case 2: The entity is a part of an archive.
    part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)
    if part_info and part_info.master_entity_id:
        # Check if the master it points to is still valid.
        master_manifest = await transaction.get_component(part_info.master_entity_id, SplitArchiveManifestComponent)
        if master_manifest:
            return True

    return False


@system(on_command=UnbindSplitArchiveCommand)
async def unbind_split_archive_handler(
    cmd: UnbindSplitArchiveCommand,
    transaction: WorldTransaction,
):
    """
    Handles unbinding a split archive, starting from either a master or a part.
    """
    master_entity_id: Optional[int] = None

    # Determine the master entity ID
    # Case 1: The command is run on the master entity itself.
    manifest = await transaction.get_component(cmd.entity_id, SplitArchiveManifestComponent)
    if manifest:
        master_entity_id = cmd.entity_id
    # Case 2: The command is run on a part entity.
    else:
        part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)
        if part_info and part_info.master_entity_id:
            master_entity_id = part_info.master_entity_id

    if not master_entity_id:
        logger.warning("Entity %s is not a split archive master or part. Nothing to unbind.", cmd.entity_id)
        return

    logger.info("Unbinding split archive master entity %s.", master_entity_id)

    # Get the manifest component of the actual master
    master_manifest = await transaction.get_component(master_entity_id, SplitArchiveManifestComponent)
    if not master_manifest:
        # This case could happen if the master was deleted but the part component remained.
        logger.error(
            "Could not find a split archive manifest for master entity %s, though part %s referenced it.",
            master_entity_id,
            cmd.entity_id,
        )
        return

    # Delete part info from all parts associated with the master
    stmt = select(SplitArchivePartInfoComponent).where(
        SplitArchivePartInfoComponent.master_entity_id == master_entity_id
    )
    result = await transaction.session.execute(stmt)
    parts_to_unbind = result.scalars().all()
    for part_info in parts_to_unbind:
        await transaction.remove_component(part_info)

    # Delete the manifest from the master
    await transaction.remove_component(master_manifest)

    logger.info("Successfully unbound archive for master entity %s.", master_entity_id)
