import logging
from typing import Annotated

from dam.core.systems import handles_command, listens_for
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam_archive.main import open_archive
from dam_archive.models import ArchiveMemberComponent
from dam_fs.events import FileStored
from dam_fs.functions import file_operations as dam_fs_file_operations
from dam_fs.resources import FileStorageResource
from dam_source.functions import import_functions

from ..commands import IngestAssetsCommand, IngestAssetStreamCommand

logger = logging.getLogger(__name__)


@handles_command(IngestAssetStreamCommand)
async def ingest_asset_stream_command_handler(
    cmd: IngestAssetStreamCommand,
    world: Annotated[World, "Resource"],
    transaction: EcsTransaction,
    fs_resource: Annotated[FileStorageResource, "Resource"],
):
    """
    Handles the command to ingest an in-memory asset stream.
    """
    logger.info(f"Received asset ingestion request for '{cmd.original_filename}' in world '{cmd.world_name}'.")

    try:
        cmd.file_content.seek(0)
        size_bytes = len(cmd.file_content.getvalue())

        entity = await import_functions.import_stream(
            world=world,
            transaction=transaction,
            file_content=cmd.file_content,
            original_filename=cmd.original_filename,
            size_bytes=size_bytes,
        )
        await transaction.flush()

        # TODO: The FileStored event needs a file_id, which is not directly available
        # from the import_stream function. This needs to be resolved.
        # For now, we will not fire the event.
        logger.info(
            f"Successfully processed IngestAssetStreamCommand for {cmd.original_filename}, created Entity {entity.id}"
        )

    except Exception as e:
        logger.error(
            f"Failed during ingestion request for entity {cmd.entity.id} ('{cmd.original_filename}'): {e}",
            exc_info=True,
        )
        # The scheduler will handle rollback on exception.
        raise


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


@listens_for(FileStored)
async def asset_dispatcher_system(
    event: FileStored,
    world: Annotated[World, "Resource"],
):
    """
    Listens for when a file has been stored and dispatches it to the
    appropriate processing pipeline based on its MIME type.
    """
    import magic

    logger.info(f"Dispatching asset for entity {event.entity.id} with file path '{event.file_path}'.")

    try:
        mime_type = magic.from_file(str(event.file_path), mime=True)
        logger.info(f"Detected MIME type '{mime_type}' for entity {event.entity.id}.")

        if mime_type.startswith("image/"):
            from dam_media_image.events import ImageAssetDetected

            await world.send_event(ImageAssetDetected(entity=event.entity, file_id=event.file_id))
            logger.info(f"Dispatched entity {event.entity.id} to image processing pipeline.")

        else:
            logger.info(
                f"No specific processing pipeline found for MIME type '{mime_type}' on entity {event.entity.id}."
            )

    except Exception as e:
        logger.error(
            f"Failed during asset dispatch for entity {event.entity.id} ('{event.file_path}'): {e}",
            exc_info=True,
        )
        raise
