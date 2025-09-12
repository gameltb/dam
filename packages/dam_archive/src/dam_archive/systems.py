import logging
from typing import Annotated, BinaryIO, List, Optional

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.models.metadata.mime_type_component import MimeTypeComponent
from sqlalchemy import select

from .commands import (
    ClearArchiveComponentsCommand,
    ExtractArchiveMembersCommand,
    SetArchivePasswordCommand,
)
from .exceptions import InvalidPasswordError, PasswordRequiredError
from .main import open_archive
from .models import ArchiveInfoComponent, ArchiveMemberComponent, ArchivePasswordComponent

logger = logging.getLogger(__name__)


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
    archive_member_component = await transaction.get_component(cmd.entity_id, ArchiveMemberComponent)
    if not archive_member_component:
        return None

    target_entity_id = archive_member_component.archive_entity_id
    path_in_archive = archive_member_component.path_in_archive
    password_comp = await transaction.get_component(target_entity_id, ArchivePasswordComponent)
    password = password_comp.password if password_comp else None

    archive_stream_cmd = GetAssetStreamCommand(entity_id=target_entity_id)
    archive_stream_result = await world.dispatch_command(archive_stream_cmd)
    archive_stream = archive_stream_result.get_first_ok_value()

    if not archive_stream:
        logger.warning(f"Could not get stream for parent archive {target_entity_id}")
        return None

    mime_type_comp = await transaction.get_component(target_entity_id, MimeTypeComponent)
    if not mime_type_comp:
        logger.warning(f"Could not get mime type for parent archive {target_entity_id}")
        return None

    try:
        archive = open_archive(archive_stream, mime_type_comp.value, password)
        if archive:
            return archive.open_file(path_in_archive)
    except Exception as e:
        logger.error(f"Failed to open member '{path_in_archive}' from archive stream for entity {cmd.entity_id}: {e}")

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


@system(on_command=ExtractArchiveMembersCommand)
async def extract_archive_members_handler(
    cmd: ExtractArchiveMembersCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handles processing an archive, ingesting its members, and linking them.
    """
    logger.info(f"Processing archive for entity {cmd.entity_id}")

    archive_stream_cmd = GetAssetStreamCommand(entity_id=cmd.entity_id)
    archive_stream_result = await world.dispatch_command(archive_stream_cmd)
    archive_stream = archive_stream_result.get_first_ok_value()

    if not archive_stream:
        logger.error(f"Could not get stream for archive entity {cmd.entity_id}")
        return

    mime_type_comp = await transaction.get_component(cmd.entity_id, MimeTypeComponent)
    if not mime_type_comp:
        logger.error(f"Could not get mime type for archive entity {cmd.entity_id}")
        return

    stored_password_comp = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
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
            logger.info(f"Successfully opened archive {cmd.entity_id} with password: {'yes' if pwd else 'no'}")
            break
        except InvalidPasswordError:
            if temp_archive:
                temp_archive.close()
            continue
        except (IOError, RuntimeError) as e:
            if temp_archive:
                temp_archive.close()
            logger.error(f"Failed to open archive {cmd.entity_id}: {e}")
            return

    if not archive:
        raise PasswordRequiredError(f"A password is required for archive entity {cmd.entity_id}")

    try:
        if correct_password and (not stored_password or correct_password != stored_password):
            await world.dispatch_command(SetArchivePasswordCommand(entity_id=cmd.entity_id, password=correct_password))

        total_size = sum(m.size for m in archive.list_files())
        if cmd.init_progress_callback:
            cmd.init_progress_callback(total_size)

        for member_file in archive.iter_files():
            try:
                with member_file as member_stream:
                    print(member_file.name)
                    get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=member_stream)
                    command_result = await world.dispatch_command(get_or_create_cmd)
                    member_entity, _ = command_result.get_one_value()

                    member_comp = ArchiveMemberComponent(
                        archive_entity_id=cmd.entity_id,
                        path_in_archive=member_file.name,
                    )
                    await transaction.add_component_to_entity(member_entity.id, member_comp)

                if cmd.update_progress_callback:
                    cmd.update_progress_callback(member_file.size)
            except Exception as e:
                logger.error(f"Failed to process member '{member_file.name}' from archive {cmd.entity_id}: {e}")
                if cmd.error_callback:
                    if not cmd.error_callback(member_file.name, e):
                        break

        info_comp = ArchiveInfoComponent(file_count=len(archive.list_files()))
        await transaction.add_component_to_entity(cmd.entity_id, info_comp)

        logger.info(f"Finished processing archive {cmd.entity_id}, processed {len(archive.list_files())} members.")
    finally:
        if archive:
            archive.close()


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
