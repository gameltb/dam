import logging
from typing import Any, cast, Dict

from dam.core.commands import AddHashesFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
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
        self.entity_id = entity_id
        self.algorithm = algorithm
        self.existing_hash = existing_hash
        self.new_hash = new_hash
        super().__init__(
            f"Hash mismatch for entity {entity_id} with algorithm {algorithm.name}: "
            f"existing {existing_hash} != new {new_hash}"
        )


async def add_hash_component(
    transaction: EcsTransaction,
    entity_id: int,
    algorithm: HashAlgorithm,
    hash_value: bytes,
) -> None:
    """Adds a single hash component to an entity."""
    component_class = HASH_ALGORITHM_TO_COMPONENT.get(algorithm)
    if not component_class:
        logger.warning(f"No component class found for hash algorithm {algorithm.name}. Skipping.")
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
        logger.info(f"Added {component_class.__name__} to entity {entity_id}")


async def add_hashes_to_entity(
    transaction: EcsTransaction,
    entity_id: int,
    hashes: Dict[HashAlgorithm, Any],
) -> None:
    """
    Adds multiple hash components to an entity from a dictionary of hashes.
    """
    for algorithm, hash_value in hashes.items():
        # Special handling for CRC32 as it's an int
        if algorithm == HashAlgorithm.CRC32:
            hash_value_bytes = hash_value.to_bytes(4, "big")
        else:
            hash_value_bytes = hash_value

        await add_hash_component(
            transaction=transaction,
            entity_id=entity_id,
            algorithm=algorithm,
            hash_value=hash_value_bytes,
        )


@system(on_command=AddHashesFromStreamCommand)
async def add_hashes_from_stream_system(cmd: AddHashesFromStreamCommand, transaction: EcsTransaction) -> None:
    """
    Handles the command to calculate and add multiple hash components to an entity from a stream.
    """
    logger.info(f"System handling AddHashesFromStreamCommand for entity {cmd.entity_id}")

    hashes = calculate_hashes_from_stream(cmd.stream, cmd.algorithms)

    for algorithm, hash_value in hashes.items():
        # Special handling for CRC32 as it's an int
        if algorithm == HashAlgorithm.CRC32:
            hash_value_bytes = hash_value.to_bytes(4, "big")  # type: ignore
        else:
            hash_value_bytes = hash_value  # type: ignore

        await add_hash_component(
            transaction=transaction,
            entity_id=cmd.entity_id,
            algorithm=algorithm,
            hash_value=hash_value_bytes,
        )

    await transaction.flush()
