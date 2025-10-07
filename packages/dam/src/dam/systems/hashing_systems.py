"""Systems for calculating and storing content hashes."""

import logging
from typing import Any, cast

from dam.commands.hashing_commands import AddHashesFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.models.hashes import (
    ContentHashBLAKE3Component,
    ContentHashCRC32Component,
    ContentHashMD5Component,
    ContentHashSHA1Component,
    ContentHashSHA256Component,
)
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream

logger = logging.getLogger(__name__)

HASH_ALGORITHM_TO_COMPONENT = {
    HashAlgorithm.MD5: ContentHashMD5Component,
    HashAlgorithm.SHA1: ContentHashSHA1Component,
    HashAlgorithm.SHA256: ContentHashSHA256Component,
    HashAlgorithm.CRC32: ContentHashCRC32Component,
    HashAlgorithm.BLAKE3: ContentHashBLAKE3Component,
}


class HashMismatchError(Exception):
    """Raised when a calculated hash does not match an existing hash component."""

    def __init__(self, entity_id: int, algorithm: HashAlgorithm, existing_hash: str, new_hash: str):
        """Initialize the error."""
        self.entity_id = entity_id
        self.algorithm = algorithm
        self.existing_hash = existing_hash
        self.new_hash = new_hash
        super().__init__(
            f"Hash mismatch for entity {entity_id} with algorithm {algorithm.name}: "
            f"existing {existing_hash} != new {new_hash}"
        )


async def add_hash_component(
    transaction: WorldTransaction,
    entity_id: int,
    algorithm: HashAlgorithm,
    hash_value: bytes,
) -> None:
    """Add a single hash component to an entity."""
    component_class = HASH_ALGORITHM_TO_COMPONENT.get(algorithm)
    if not component_class:
        logger.warning("No component class found for hash algorithm %s. Skipping.", algorithm.name)
        return

    existing_component = await transaction.get_component(entity_id, component_class)

    if existing_component:
        existing_component_any = cast(Any, existing_component)
        if existing_component_any.hash_value != hash_value:
            raise HashMismatchError(
                entity_id=entity_id,
                algorithm=algorithm,
                existing_hash=existing_component_any.hash_value.hex(),
                new_hash=hash_value.hex(),
            )
    else:
        new_component = component_class(hash_value=hash_value)
        await transaction.add_component_to_entity(entity_id, new_component)
        logger.info("Added %s to entity %s", component_class.__name__, entity_id)


async def add_hashes_to_entity(
    transaction: WorldTransaction,
    entity_id: int,
    hashes: dict[HashAlgorithm, Any],
) -> None:
    """Add multiple hash components to an entity from a dictionary of hashes."""
    for algorithm, hash_value in hashes.items():
        # Special handling for CRC32 as it's an int
        hash_value_bytes = hash_value.to_bytes(4, "big") if algorithm == HashAlgorithm.CRC32 else hash_value

        await add_hash_component(
            transaction=transaction,
            entity_id=entity_id,
            algorithm=algorithm,
            hash_value=hash_value_bytes,
        )


@system(on_command=AddHashesFromStreamCommand)
async def add_hashes_from_stream_system(cmd: AddHashesFromStreamCommand, transaction: WorldTransaction) -> None:
    """Handle the command to calculate and add multiple hash components to an entity from a stream."""
    logger.info("System handling AddHashesFromStreamCommand for entity %s", cmd.entity_id)

    async with cmd.stream_provider.get_stream() as stream:
        hashes = calculate_hashes_from_stream(stream, cmd.algorithms)

    for algorithm, hash_value in hashes.items():
        hash_value_bytes: bytes
        if isinstance(hash_value, int):
            hash_value_bytes = hash_value.to_bytes(4, "big")
        elif isinstance(hash_value, bytes):
            hash_value_bytes = hash_value
        else:
            logger.warning("Unsupported hash value type for %s: %s", algorithm, type(hash_value))
            continue

        await add_hash_component(
            transaction=transaction,
            entity_id=cmd.entity_id,
            algorithm=algorithm,
            hash_value=hash_value_bytes,
        )

    await transaction.flush()
