from io import BytesIO

import pytest

from dam.core.commands import AddHashesFromStreamCommand
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
    async with world.db_session_maker() as session:
        entity = await ecs_functions.create_entity(session)
        await session.commit()
        entity_id = entity.id

    data = b"hello world"
    stream = BytesIO(data)
    command = AddHashesFromStreamCommand(
        entity_id=entity_id,
        stream=stream,
        algorithms={HashAlgorithm.MD5, HashAlgorithm.SHA256},
    )
    await world.dispatch_command(command)

    async with world.db_session_maker() as session:
        md5_comp = await ecs_functions.get_component(session, entity_id, ContentHashMD5Component)
        assert md5_comp is not None
        import hashlib

        assert md5_comp.hash_value == hashlib.md5(data).digest()

        sha256_comp = await ecs_functions.get_component(session, entity_id, ContentHashSHA256Component)
        assert sha256_comp is not None
        assert sha256_comp.hash_value == hashlib.sha256(data).digest()


@pytest.mark.asyncio
async def test_add_hashes_from_stream_system_captures_mismatch_error(test_world_alpha: World) -> None:
    """
    Tests that the system returns an Err result if a hash already exists and does not match.
    """
    world = test_world_alpha
    async with world.db_session_maker() as session:
        entity = await ecs_functions.create_entity(session)
        # Add a component with a wrong hash.
        wrong_hash_comp = ContentHashMD5Component(hash_value=b"a" * 16)
        await ecs_functions.add_component_to_entity(session, entity.id, wrong_hash_comp)
        await session.commit()
        entity_id = entity.id

    data = b"hello world"
    stream = BytesIO(data)
    command = AddHashesFromStreamCommand(
        entity_id=entity_id,
        stream=stream,
        algorithms={HashAlgorithm.MD5},
    )

    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 1
    handler_res = result.results[0]

    assert handler_res.is_err()
    assert isinstance(handler_res.exception, HashMismatchError)
