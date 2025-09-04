import logging
from typing import Annotated

from dam.core.commands import AddHashesFromStreamCommand, GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.core.entity import Entity
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream

logger = logging.getLogger(__name__)


@system(on_command=GetOrCreateEntityFromStreamCommand)
async def get_or_create_entity_from_stream_handler(
    cmd: GetOrCreateEntityFromStreamCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> tuple[Entity, bytes]:
    """
    Handles getting or creating an entity from a stream.
    """
    logger.info(f"System handling GetOrCreateEntityFromStreamCommand in world {world.name}")
    try:
        cmd.stream.seek(0)
        hashes = calculate_hashes_from_stream(cmd.stream, {HashAlgorithm.SHA256})
        sha256_bytes = hashes[HashAlgorithm.SHA256]
    except (IOError, FileNotFoundError) as e:
        raise  # Re-raise the exception to be handled by the caller

    existing_entity = await transaction.find_entity_by_content_hash(sha256_bytes, "sha256")
    entity = None

    if existing_entity:
        entity = existing_entity
        logger.info(f"Content already exists as Entity ID {entity.id}.")
    else:
        entity = await ecs_functions.create_entity(transaction.session)
        logger.info(f"Creating new Entity ID {entity.id}.")

    if not entity:
        raise Exception("Failed to create or find entity for the asset.")

    # Dispatch command to add all hashes
    cmd.stream.seek(0)
    add_hashes_command = AddHashesFromStreamCommand(
        entity_id=entity.id,
        stream=cmd.stream,
        algorithms={HashAlgorithm.MD5, HashAlgorithm.SHA256},
    )
    await world.dispatch_command(add_hashes_command)

    from dam_app.commands import ExtractMetadataCommand

    await world.dispatch_command(ExtractMetadataCommand(entity_id=entity.id))

    return entity, sha256_bytes
