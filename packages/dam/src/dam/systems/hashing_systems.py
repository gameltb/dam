import logging

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


@system(on_command=AddHashesFromStreamCommand)
async def add_hashes_from_stream_system(cmd: AddHashesFromStreamCommand, transaction: EcsTransaction) -> None:
    """
    Handles the command to calculate and add multiple hash components to an entity from a stream.
    """
    logger.info(f"System handling AddHashesFromStreamCommand for entity {cmd.entity_id}")

    hashes = calculate_hashes_from_stream(cmd.stream, cmd.algorithms)

    for algorithm, hash_value in hashes.items():
        component_class = HASH_ALGORITHM_TO_COMPONENT.get(algorithm)
        if not component_class:
            logger.warning(f"No component class found for hash algorithm {algorithm.name}. Skipping.")
            continue

        existing_component = await transaction.get_component(cmd.entity_id, component_class)

        # Special handling for CRC32 as it's an int
        if algorithm == HashAlgorithm.CRC32:
            hash_value_bytes = hash_value.to_bytes(4, "big")  # type: ignore
        else:
            hash_value_bytes = hash_value  # type: ignore

        if existing_component:
            if getattr(existing_component, "hash_value") != hash_value_bytes:
                raise HashMismatchError(
                    entity_id=cmd.entity_id,
                    algorithm=algorithm,
                    existing_hash=getattr(existing_component, "hash_value").hex(),
                    new_hash=hash_value_bytes.hex(),
                )
        else:
            new_component = component_class(hash_value=hash_value_bytes)  # type: ignore
            await transaction.add_component_to_entity(cmd.entity_id, new_component)
            logger.info(f"Added {component_class.__name__} to entity {cmd.entity_id}")

    await transaction.flush()
