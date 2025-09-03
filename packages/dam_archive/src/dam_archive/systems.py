import asyncio
import logging
from typing import Annotated

from dam.core.systems import handles_command
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam_fs.commands import GetAssetStreamCommand
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from .commands import IngestAssetsCommand, SetArchivePasswordCommand
from .main import open_archive
from .models import ArchiveMemberComponent, ArchivePasswordComponent

logger = logging.getLogger(__name__)


@handles_command(SetArchivePasswordCommand)
async def set_archive_password_handler(
    cmd: SetArchivePasswordCommand,
    transaction: EcsTransaction,
):
    """
    Handles setting the password for an archive.
    """
    password_comp = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    if password_comp:
        password_comp.password = cmd.password
    else:
        password_comp = ArchivePasswordComponent(entity_id=cmd.entity_id, password=cmd.password)
        await transaction.add_component_to_entity(cmd.entity_id, password_comp)


from dam.functions import ecs_functions
from dam_fs.functions import file_operations as dam_fs_file_operations


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

    password_comp = await transaction.get_component(target_entity_id, ArchivePasswordComponent)
    passwords = [password_comp.password] if password_comp else []

    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            if potential_path and await asyncio.to_thread(potential_path.is_file):
                archive = open_archive(str(potential_path), passwords)
                if archive:
                    return archive.open_file(path_in_archive)
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Could not resolve or find file for URL '{loc.url}' for entity {target_entity_id}: {e}")
            continue

    return None  # No valid local file found for the archive


@handles_command(IngestAssetsCommand)
async def asset_ingestion_system(
    cmd: IngestAssetsCommand,
    world: Annotated[World, "Resource"],
    transaction: EcsTransaction,
):
    """
    Handles the command to ingest assets from a list of file paths.
    This system expands archives and creates entities for all files.
    """
    logger.info(f"Received asset ingestion request for {len(cmd.file_paths)} files.")
    entity_ids = []

    for file_path in cmd.file_paths:
        try:
            # For now, we only support one password for all archives.
            # A more advanced implementation could try multiple passwords.
            password = cmd.passwords[0] if cmd.passwords else None
            archive = open_archive(file_path, password)
            if archive:
                # This is an archive file, create an entity for the archive itself
                archive_entity = await dam_fs_file_operations.create_entity_with_file(
                    transaction, world.world_config, file_path
                )
                entity_ids.append(archive_entity.id)

                # Now, create entities for each file within the archive
                for member_name in archive.list_files():
                    member_entity = await ecs_functions.create_entity(transaction.session)
                    await transaction.add_component_to_entity(
                        member_entity.id,
                        ArchiveMemberComponent(
                            archive_entity_id=archive_entity.id,
                            path_in_archive=member_name,
                        ),
                    )
                    entity_ids.append(member_entity.id)
            else:
                # This is a regular file
                entity = await dam_fs_file_operations.create_entity_with_file(
                    transaction, world.world_config, file_path
                )
                entity_ids.append(entity.id)

        except Exception as e:
            logger.error(f"Failed to ingest asset from '{file_path}': {e}", exc_info=True)

    return entity_ids
