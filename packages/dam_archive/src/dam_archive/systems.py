import logging
from typing import IO, Annotated, List, Optional

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World

from .commands import ExtractArchiveMembersCommand, SetArchivePasswordCommand
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
) -> Optional[IO[bytes]]:
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

    # Get asset filenames for the parent archive
    filenames_cmd = GetAssetFilenamesCommand(entity_id=target_entity_id)
    filenames_result = await world.dispatch_command(filenames_cmd)
    filenames = filenames_result.get_first_ok_value()
    filename = filenames[0] if filenames else None

    if not filename:
        logger.warning(f"Could not get filename for parent archive {target_entity_id}")
        return None

    try:
        archive = open_archive(archive_stream, filename, password)
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
    archive_member_comp = await transaction.get_component(cmd.entity_id, ArchiveMemberComponent)
    if archive_member_comp and archive_member_comp.path_in_archive:
        return [archive_member_comp.path_in_archive]
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

    # Get asset stream
    archive_stream_cmd = GetAssetStreamCommand(entity_id=cmd.entity_id)
    archive_stream_result = await world.dispatch_command(archive_stream_cmd)
    archive_stream = archive_stream_result.get_first_ok_value()

    if not archive_stream:
        logger.error(f"Could not get stream for archive entity {cmd.entity_id}")
        return

    # Get asset filenames
    filenames_cmd = GetAssetFilenamesCommand(entity_id=cmd.entity_id)
    filenames_result = await world.dispatch_command(filenames_cmd)
    filenames = filenames_result.get_first_ok_value()
    filename = filenames[0] if filenames else None

    if not filename:
        logger.error(f"Could not get filename for archive entity {cmd.entity_id}")
        return

    # Prioritize stored password
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
        try:
            archive_stream.seek(0)
            archive = open_archive(archive_stream, filename, pwd)
            if not archive:
                logger.error("Could not find a handler for the archive type.")
                return

            # Test the password by trying to list files
            archive.list_files()
            correct_password = pwd
            logger.info(f"Successfully opened archive {cmd.entity_id} with password: {'yes' if pwd else 'no'}")
            break
        except InvalidPasswordError:
            archive = None
            continue
        except (IOError, RuntimeError) as e:
            logger.error(f"Failed to open archive {cmd.entity_id}: {e}")
            return

    if not archive:
        raise PasswordRequiredError(f"A password is required for archive entity {cmd.entity_id}")

    # Save the correct password if we found a new one
    if correct_password and (not stored_password or correct_password != stored_password):
        await world.dispatch_command(SetArchivePasswordCommand(entity_id=cmd.entity_id, password=correct_password))

    member_files = archive.list_files()
    total_size = sum(m.size for m in member_files)
    if cmd.init_progress_callback:
        cmd.init_progress_callback(total_size)

    for member in member_files:
        try:
            with archive.open_file(member.name) as member_stream:
                get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=member_stream)
                command_result = await world.dispatch_command(get_or_create_cmd)
                member_entity, _ = command_result.get_one_value()

                member_comp = ArchiveMemberComponent(
                    archive_entity_id=cmd.entity_id,
                    path_in_archive=member.name,
                )
                await transaction.add_component_to_entity(member_entity.id, member_comp)

            if cmd.update_progress_callback:
                cmd.update_progress_callback(member.size)
        except Exception as e:
            logger.error(f"Failed to process member '{member.name}' from archive {cmd.entity_id}: {e}")
            if cmd.error_callback:
                if not cmd.error_callback(member.name, e):
                    break
            # If no error callback, continue by default

    info_comp = ArchiveInfoComponent(file_count=len(member_files))
    await transaction.add_component_to_entity(cmd.entity_id, info_comp)

    logger.info(f"Finished processing archive {cmd.entity_id}, processed {len(member_files)} members.")
