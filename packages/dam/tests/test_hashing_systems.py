import hashlib
from io import BytesIO

import pytest

from dam.commands.hashing_commands import AddHashesFromStreamCommand
from dam.core.transaction import WorldTransaction
from dam.core.types import CallableStreamProvider
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.hashes import (
    ContentHashMD5Component,
    ContentHashSHA256Component,
)
from dam.systems.hashing_systems import HashMismatchError
from dam.utils.hash_utils import HashAlgorithm


@pytest.mark.asyncio
async def test_add_hashes_from_stream_system(test_world_alpha: World) -> None:
    """
    Tests that the add_hashes_from_stream_system correctly adds hash components.
    """
    world = test_world_alpha
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        entity = await ecs_functions.create_entity(transaction.session)
        await transaction.session.commit()
        entity_id = entity.id

    data = b"hello world"
    stream = BytesIO(data)
    stream_provider = CallableStreamProvider(lambda: stream)
    command = AddHashesFromStreamCommand(
        entity_id=entity_id,
        stream_provider=stream_provider,
        algorithms={HashAlgorithm.MD5, HashAlgorithm.SHA256},
    )
    await world.dispatch_command(command).get_all_results()

    async with tm() as transaction:
        md5_comp = await ecs_functions.get_component(transaction.session, entity_id, ContentHashMD5Component)
        assert md5_comp is not None
        assert md5_comp.hash_value == hashlib.md5(data).digest()

        sha256_comp = await ecs_functions.get_component(transaction.session, entity_id, ContentHashSHA256Component)
        assert sha256_comp is not None
        assert sha256_comp.hash_value == hashlib.sha256(data).digest()


@pytest.mark.asyncio
async def test_add_hashes_from_stream_system_propagates_mismatch_error(
    test_world_alpha: World,
) -> None:
    """
    Tests that the system propagates a HashMismatchError if a hash already exists and does not match.
    """
    world = test_world_alpha
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        entity = await ecs_functions.create_entity(transaction.session)
        # Add a component with a wrong hash.
        wrong_hash_comp = ContentHashMD5Component(hash_value=b"a" * 16)
        await ecs_functions.add_component_to_entity(transaction.session, entity.id, wrong_hash_comp)
        await transaction.session.commit()
        entity_id = entity.id

    data = b"hello world"
    stream = BytesIO(data)
    stream_provider = CallableStreamProvider(lambda: stream)
    command = AddHashesFromStreamCommand(
        entity_id=entity_id,
        stream_provider=stream_provider,
        algorithms={HashAlgorithm.MD5},
    )

    with pytest.raises(HashMismatchError):
        await world.dispatch_command(command).get_all_results()
