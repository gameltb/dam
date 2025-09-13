import io
import logging
from typing import Annotated, BinaryIO, List, Optional, cast

from dam.commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    SetMimeTypeFromBufferCommand,
)
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.models.metadata.mime_type_component import MimeTypeComponent
from dam.utils.stream_utils import ChainedStream
from dam_fs.models import FilePropertiesComponent
from sqlalchemy import select

from . import split_detector
from .commands import (
    ClearArchiveComponentsCommand,
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    IngestArchiveMembersCommand,
    SetArchivePasswordCommand,
    UnbindSplitArchiveCommand,
)
from .exceptions import InvalidPasswordError, PasswordRequiredError
from .main import open_archive
from .models import (
    ArchiveInfoComponent,
    ArchiveMemberComponent,
    ArchivePasswordComponent,
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)

logger = logging.getLogger(__name__)


@system(on_command=DiscoverAndBindCommand)
async def discover_and_bind_handler(
    cmd: DiscoverAndBindCommand,
    transaction: EcsTransaction,
):
    """
    Scans given paths, tags archive parts, and assembles complete sets.
    """
    import os
    from pathlib import Path

    logger.info(f"Starting discovery and binding for paths: {cmd.paths}")

    # 1. Collect all filenames from the given paths
    target_filenames = set()
    for p_str in cmd.paths:
        p = Path(p_str)
        if not p.exists():
            logger.warning(f"Path does not exist, skipping: {p}")
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                for name in files:
                    target_filenames.add(os.path.join(root, name))
        else:
            target_filenames.add(str(p))

    # 2. Find entities corresponding to these filenames
    stmt = select(FilePropertiesComponent).where(
        FilePropertiesComponent.original_filename.in_(target_filenames)
    )
    result = await transaction.session.execute(stmt)
    file_props_comps = result.scalars().all()

    logger.info(f"Found {len(file_props_comps)} entities in the world matching the given paths.")

    # 3. Tag all found entities if they are split archive parts
    for props in file_props_comps:
        split_info = split_detector.detect(props.original_filename)
        if split_info:
            existing_comp = await transaction.get_component(props.entity_id, SplitArchivePartInfoComponent)
            if not existing_comp:
                logger.info(f"Tagging '{props.original_filename}' as a split archive part.")
                part_info_comp = SplitArchivePartInfoComponent(
                    base_name=split_info.base_name, part_num=split_info.part_num
                )
                await transaction.add_component_to_entity(props.entity_id, part_info_comp)

    # 4. Find all untagged parts and group them
    stmt_untagged = select(SplitArchivePartInfoComponent).where(
        SplitArchivePartInfoComponent.master_entity_id.is_(None)
    )
    result_untagged = await transaction.session.execute(stmt_untagged)
    untagged_parts = result_untagged.scalars().all()

    parts_by_basename = {}
    for part in untagged_parts:
        parts_by_basename.setdefault(part.base_name, []).append(part)

    # 5. Assemble complete groups
    for base_name, parts in parts_by_basename.items():
        parts_by_num = {p.part_num: p for p in parts}
        max_part_num = max(parts_by_num.keys())
        is_complete = all(i in parts_by_num for i in range(1, max_part_num + 1))

        if is_complete:
            logger.info(f"Found complete split archive '{base_name}' with {max_part_num} parts. Assembling.")

            master_entity = await transaction.create_entity()
            await transaction.add_component_to_entity(
                master_entity.id,
                FilePropertiesComponent(original_filename=f"{base_name} (Split Archive)"),
            )

            sorted_part_ids = [parts_by_num[i].entity_id for i in range(1, max_part_num + 1)]
            manifest = SplitArchiveManifestComponent(part_entity_ids=sorted_part_ids)
            await transaction.add_component_to_entity(master_entity.id, manifest)

            first_part_entity_id = sorted_part_ids[0]
            mime_type_comp = await transaction.get_component(first_part_entity_id, MimeTypeComponent)
            if mime_type_comp:
                await transaction.add_component_to_entity(master_entity.id, MimeTypeComponent(value=mime_type_comp.value))

            for part_comp in parts:
                part_comp.master_entity_id = master_entity.id
        else:
            logger.info(f"Split archive '{base_name}' is incomplete. Skipping assembly.")


@system(on_command=CreateMasterArchiveCommand)
async def create_master_archive_handler(
    cmd: CreateMasterArchiveCommand,
    transaction: EcsTransaction,
):
    """
    Handles the manual creation of a master entity for a split archive.
    """
    logger.info(f"Manually creating master archive '{cmd.name}' for {len(cmd.part_entity_ids)} parts.")

    # 1. Create master entity and its components
    master_entity = await transaction.create_entity()
    await transaction.add_component_to_entity(
        master_entity.id,
        FilePropertiesComponent(original_filename=cmd.name),
    )
    manifest = SplitArchiveManifestComponent(part_entity_ids=cmd.part_entity_ids)
    await transaction.add_component_to_entity(master_entity.id, manifest)

    # 2. Copy mime type from first part
    if cmd.part_entity_ids:
        first_part_id = cmd.part_entity_ids[0]
        mime_type_comp = await transaction.get_component(first_part_id, MimeTypeComponent)
        if mime_type_comp:
            await transaction.add_component_to_entity(master_entity.id, MimeTypeComponent(value=mime_type_comp.value))

    # 3. Update all part components to link to the new master
    for part_id in cmd.part_entity_ids:
        file_props = await transaction.get_component(part_id, FilePropertiesComponent)
        if not file_props:
            logger.warning(f"Skipping part entity {part_id} as it has no file properties.")
            continue

        split_info = split_detector.detect(file_props.original_filename)
        if not split_info:
            logger.warning(f"Skipping part entity {part_id} as it does not look like a split archive part.")
            continue

        part_info_comp = await transaction.get_component(part_id, SplitArchivePartInfoComponent)
        if part_info_comp:
            part_info_comp.master_entity_id = master_entity.id
        else:
            new_part_info = SplitArchivePartInfoComponent(
                base_name=split_info.base_name,
                part_num=split_info.part_num,
                master_entity_id=master_entity.id,
            )
            await transaction.add_component_to_entity(part_id, new_part_info)

    logger.info(f"Successfully created master entity {master_entity.id} for archive '{cmd.name}'.")


@system(on_command=UnbindSplitArchiveCommand)
async def unbind_split_archive_handler(
    cmd: UnbindSplitArchiveCommand,
    transaction: EcsTransaction,
):
    """
    Handles unbinding a split archive.
    """
    logger.info(f"Unbinding split archive master entity {cmd.master_entity_id}.")

    manifest = await transaction.get_component(cmd.master_entity_id, SplitArchiveManifestComponent)
    if not manifest:
        logger.warning(f"No split archive manifest found for entity {cmd.master_entity_id}. Nothing to unbind.")
        return

    # Delete part info from all parts
    for part_id in manifest.part_entity_ids:
        part_info = await transaction.get_component(part_id, SplitArchivePartInfoComponent)
        if part_info:
            await transaction.remove_component(part_info)

    # Delete the manifest from the master
    await transaction.remove_component(manifest)

    logger.info(f"Successfully unbound archive for master entity {cmd.master_entity_id}.")


@system(on_command=SetArchivePasswordCommand)
async def set_archive_password_handler(
    cmd: SetArchivePasswordCommand,
    transaction: EcsTransaction,
) -> None:
    """
    Handles setting the password for an archive.
    """
    password_comp = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    if password_comp:
        password_comp.password = cmd.password
    else:
        password_comp = ArchivePasswordComponent(password=cmd.password)
        await transaction.add_component_to_entity(cmd.entity_id, password_comp)


@system(on_command=GetAssetStreamCommand)
async def get_archive_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> Optional[BinaryIO]:
    """
    Handles getting a stream for an asset that is part of an archive.
    """
    archive_member_components = await transaction.get_components(cmd.entity_id, ArchiveMemberComponent)
    if not archive_member_components:
        return None

    for component in archive_member_components:
        target_entity_id = component.archive_entity_id
        path_in_archive = component.path_in_archive
        password_comp = await transaction.get_component(target_entity_id, ArchivePasswordComponent)
        password = password_comp.password if password_comp else None

        archive_stream_cmd = GetAssetStreamCommand(entity_id=target_entity_id)
        archive_stream_result = await world.dispatch_command(archive_stream_cmd)
        try:
            archive_stream = archive_stream_result.get_first_non_none_value()
        except ValueError:
            archive_stream = None

        if not archive_stream:
            logger.warning(f"Could not get stream for parent archive {target_entity_id}")
            continue

        mime_type_comp = await transaction.get_component(target_entity_id, MimeTypeComponent)
        if not mime_type_comp:
            logger.warning(f"Could not get mime type for parent archive {target_entity_id}")
            continue

        try:
            archive = open_archive(archive_stream, mime_type_comp.value, password)
            if archive:
                return archive.open_file(path_in_archive)
        except Exception as e:
            logger.error(
                f"Failed to open member '{path_in_archive}' from archive stream for entity {cmd.entity_id}: {e}"
            )

    return None


@system(on_command=GetAssetFilenamesCommand)
async def get_archive_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: EcsTransaction,
) -> Optional[List[str]]:
    """
    Handles getting filenames for assets that are members of an archive.
    """
    archive_member_comps = await transaction.get_components(cmd.entity_id, ArchiveMemberComponent)
    if archive_member_comps:
        return [archive_member_comp.path_in_archive for archive_member_comp in archive_member_comps]
    return None


async def _perform_ingestion(
    entity_id: int,
    archive_stream: BinaryIO,
    cmd: IngestArchiveMembersCommand,
    world: Annotated[World, "Resource"],
    transaction: EcsTransaction,
):
    """
    The core extraction logic, once an archive stream is prepared.
    """
    mime_type_comp = await transaction.get_component(entity_id, MimeTypeComponent)
    if not mime_type_comp:
        logger.error(f"Could not get mime type for archive entity {entity_id}")
        return

    stored_password_comp = await transaction.get_component(entity_id, ArchivePasswordComponent)
    stored_password = stored_password_comp.password if stored_password_comp else None

    passwords_to_try: List[Optional[str]] = [None]
    if stored_password:
        passwords_to_try.append(stored_password)
    if cmd.passwords:
        passwords_to_try.extend(p for p in cmd.passwords if p)

    archive = None
    correct_password = None

    for pwd in passwords_to_try:
        temp_archive = None
        try:
            archive_stream.seek(0)
            temp_archive = open_archive(archive_stream, mime_type_comp.value, pwd)
            if not temp_archive:
                logger.error("Could not find a handler for the archive type.")
                return
            correct_password = pwd
            archive = temp_archive
            logger.info(f"Successfully opened archive {entity_id} with password: {'yes' if pwd else 'no'}")
            break
        except InvalidPasswordError:
            if temp_archive:
                temp_archive.close()
            continue
        except (IOError, RuntimeError) as e:
            if temp_archive:
                temp_archive.close()
            logger.error(f"Failed to open archive {entity_id}: {e}")
            return

    if not archive:
        raise PasswordRequiredError(f"A password is required for archive entity {entity_id}")

    try:
        if correct_password and (not stored_password or correct_password != stored_password):
            await world.dispatch_command(SetArchivePasswordCommand(entity_id=entity_id, password=correct_password))

        total_size = sum(m.size for m in archive.list_files())
        if cmd.init_progress_callback:
            cmd.init_progress_callback(total_size)

        for member_file in archive.iter_files():
            try:
                with member_file as member_stream:
                    header = member_stream.read(4096)
                    full_stream = ChainedStream([io.BytesIO(header), member_stream])
                    get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=cast(BinaryIO, full_stream))
                    command_result = await world.dispatch_command(get_or_create_cmd)
                    member_entity, _ = command_result.get_one_value()
                    set_mime_cmd = SetMimeTypeFromBufferCommand(entity_id=member_entity.id, buffer=header)
                    await world.dispatch_command(set_mime_cmd)
                    member_comp = ArchiveMemberComponent(
                        archive_entity_id=entity_id, path_in_archive=member_file.name
                    )
                    await transaction.add_component_to_entity(member_entity.id, member_comp)
                if cmd.update_progress_callback:
                    cmd.update_progress_callback(member_file.size)
            except Exception as e:
                logger.error(f"Failed to process member '{member_file.name}' from archive {entity_id}: {e}")
                if cmd.error_callback:
                    if not cmd.error_callback(member_file.name, e):
                        break
        info_comp = ArchiveInfoComponent(file_count=len(archive.list_files()))
        await transaction.add_component_to_entity(entity_id, info_comp)
        logger.info(f"Finished processing archive {entity_id}, processed {len(archive.list_files())} members.")
    finally:
        if archive:
            archive.close()


@system(on_command=IngestArchiveMembersCommand)
async def ingest_archive_members_handler(
    cmd: IngestArchiveMembersCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handles processing an archive. It's the main entry point for ingestion.
    - If it's a master entity for a split archive, it ingests it.
    - If it's a part of an already assembled archive, it redirects to the master.
    - If it's a part of a non-assembled archive, it triggers assembly.
    - If it's a regular file, it ingests it.
    """
    logger.info(f"Ingestion command received for entity {cmd.entity_id}")

    # Case 1: The entity is a master entity for a split archive.
    manifest_comp = await transaction.get_component(cmd.entity_id, SplitArchiveManifestComponent)
    if manifest_comp:
        logger.info(f"Entity {cmd.entity_id} is a split archive master. Chaining part streams for ingestion.")
        part_streams = []
        try:
            for part_entity_id in manifest_comp.part_entity_ids:
                stream_cmd = GetAssetStreamCommand(entity_id=part_entity_id)
                stream_result = await world.dispatch_command(stream_cmd)
                part_stream = stream_result.get_first_non_none_value()
                if part_stream:
                    part_streams.append(part_stream)
                else:
                    raise ValueError(f"Stream for part {part_entity_id} is None")
            archive_stream = ChainedStream(part_streams)
            await _perform_ingestion(cmd.entity_id, archive_stream, cmd, world, transaction)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Could not get stream for split archive part: {e}")
            for s in part_streams:
                s.close()
        return

    # Case 2: The entity is a part of an already assembled split archive.
    part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)
    if part_info and part_info.master_entity_id:
        logger.info(
            f"Redirecting ingestion from part {cmd.entity_id} to master entity {part_info.master_entity_id}."
        )
        await world.dispatch_command(IngestArchiveMembersCommand(entity_id=part_info.master_entity_id))
        return

    # Case 3: The entity is a part of a non-assembled split archive.
    if part_info:
        raise RuntimeError(
            f"Entity {cmd.entity_id} is part of a non-assembled split archive. "
            "Please run 'discover-and-bind' or 'create-master' command first."
        )

    # Case 4: The entity is a regular, single-file archive.
    logger.info(f"Entity {cmd.entity_id} is a single-file archive. Ingesting.")
    try:
        archive_stream_cmd = GetAssetStreamCommand(entity_id=cmd.entity_id)
        archive_stream_result = await world.dispatch_command(archive_stream_cmd)
        archive_stream = archive_stream_result.get_first_non_none_value()
        if archive_stream:
            await _perform_ingestion(cmd.entity_id, archive_stream, cmd, world, transaction)
        else:
            logger.error(f"Could not get stream for single-file archive {cmd.entity_id}")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Could not get stream for single-file archive {cmd.entity_id}: {e}")


@system(on_command=ClearArchiveComponentsCommand)
async def clear_archive_components_handler(
    cmd: ClearArchiveComponentsCommand,
    transaction: EcsTransaction,
) -> None:
    """
    Handles clearing archive-related components from an entity and its members.
    """
    # Delete ArchiveInfoComponent from the main archive entity
    info_comp = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    if info_comp:
        await transaction.remove_component(info_comp)
        logger.info(f"Deleted ArchiveInfoComponent from entity {cmd.entity_id}")

    # Find all ArchiveMemberComponents that point to this archive entity
    stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    member_components = result.scalars().all()

    for member_comp in member_components:
        await transaction.remove_component(member_comp)
        logger.info(
            f"Deleted ArchiveMemberComponent from member entity {member_comp.entity_id} "
            f"(linked to archive {cmd.entity_id})"
        )
