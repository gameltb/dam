import logging
from typing import Annotated

from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam_fs.models import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from .commands import ExtractArchiveCommand
from .exceptions import PasswordRequiredError
from .main import open_archive
from .models import ArchiveInfoComponent, ArchiveMemberComponent

logger = logging.getLogger(__name__)


@system(on_command=ExtractArchiveCommand)
async def extract_archive_handler(
    cmd: ExtractArchiveCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handles extracting an archive, ingesting its members, and linking them.
    """
    logger.info(f"Extracting archive for entity {cmd.entity_id}")

    # 1. Get archive file path
    locations = await transaction.get_components(cmd.entity_id, FileLocationComponent)
    if not locations:
        logger.error(f"No file location found for archive entity {cmd.entity_id}")
        return

    archive_path = None
    for loc in locations:
        try:
            path = get_local_path_for_url(loc.url)
            if path and path.exists():
                archive_path = path
                break
        except Exception:
            continue

    if not archive_path:
        logger.error(f"Could not resolve a valid local path for archive entity {cmd.entity_id}")
        return

    # 2. Try to open the archive and extract
    archive = None
    passwords_to_try = [None] + (cmd.passwords or [])  # Always try with no password first

    for pwd in passwords_to_try:
        try:
            logger.debug(f"Attempting to open archive {cmd.entity_id} with password: {'yes' if pwd else 'no'}")
            archive = open_archive(str(archive_path), [pwd] if pwd else None)
            if not archive:
                logger.error("Could not find a handler for the archive type.")
                return

            # Test the password by trying to read the first file
            member_files = archive.list_files()
            if not member_files:
                logger.info(f"Archive {cmd.entity_id} is empty.")
                break  # Success (empty archive)

            with archive.open_file(member_files[0]):
                pass  # Just opening it is enough to trigger password error

            # If we get here, the password is correct (or no password was needed)
            logger.info(f"Successfully opened archive {cmd.entity_id} with password: {'yes' if pwd else 'no'}")
            break  # Exit the password loop

        except (IOError, RuntimeError) as e:
            if "password" in str(e).lower():
                logger.warning(f"Bad password for archive {cmd.entity_id}. Trying next.")
                archive = None  # Reset archive object
                continue
            else:
                logger.error(f"Failed to open archive {cmd.entity_id}: {e}")
                return  # Not a password error, so we fail.

    if not archive:
        raise PasswordRequiredError(f"A password is required for archive entity {cmd.entity_id}")

    # 3. Ingest members
    member_files = archive.list_files()
    for member_name in member_files:
        logger.debug(f"Processing member: {member_name}")
        with archive.open_file(member_name) as member_stream:
            # Ingest the member file from its stream
            get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=member_stream)
            command_result = await world.dispatch_command(get_or_create_cmd)
            member_entity, _ = command_result.get_one_value()

            # Link member to archive
            member_comp = ArchiveMemberComponent(
                archive_entity_id=cmd.entity_id,
                path_in_archive=member_name,
            )
            await transaction.add_component_to_entity(member_entity.id, member_comp)

    # 4. Mark archive as processed
    info_comp = ArchiveInfoComponent(file_count=len(member_files))
    await transaction.add_component_to_entity(cmd.entity_id, info_comp)

    logger.info(f"Finished extracting archive {cmd.entity_id}, processed {len(member_files)} members.")
