"""Systems for handling core entity operations."""

import io
import logging
from typing import Annotated

from dam.commands.asset_commands import GetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.core.entity import Entity
from dam.systems.hashing_systems import add_hashes_to_entity
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream

logger = logging.getLogger(__name__)


@system(on_command=GetOrCreateEntityFromStreamCommand)
async def get_or_create_entity_from_stream_handler(
    cmd: GetOrCreateEntityFromStreamCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> tuple[Entity, bytes]:
    """Handle getting or creating an entity from a stream."""
    logger.info("System handling GetOrCreateEntityFromStreamCommand in world %s", world.name)
    all_algorithms = {HashAlgorithm.MD5, HashAlgorithm.SHA256, HashAlgorithm.CRC32, HashAlgorithm.BLAKE3}

    async with cmd.stream_provider.get_stream() as stream:
        # Read the entire stream into memory to handle both seekable and non-seekable streams uniformly
        # and allow for multiple hash calculations without re-reading from the source.
        # This is a simplification; for very large files, a temporary file might be better.
        content = stream.read()

    in_memory_stream = io.BytesIO(content)
    hashes = calculate_hashes_from_stream(in_memory_stream, all_algorithms)
    sha256_bytes = hashes[HashAlgorithm.SHA256]

    existing_entity = await transaction.find_entity_by_content_hash(sha256_bytes, "sha256")  # type: ignore
    entity = None

    if existing_entity:
        entity = existing_entity
        logger.info("Content already exists as Entity ID %s.", entity.id)
    else:
        entity = await ecs_functions.create_entity(transaction.session)
        logger.info("Creating new Entity ID %s.", entity.id)

    if not entity:
        raise Exception("Failed to create or find entity for the asset.")

    # Since we have all hashes, we can add them directly.
    await add_hashes_to_entity(transaction, entity.id, hashes)
    await transaction.flush()

    return entity, sha256_bytes  # type: ignore
