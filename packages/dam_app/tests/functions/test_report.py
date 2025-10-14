"""Tests for report functions."""

import hashlib
from datetime import datetime

import pytest
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_fs.models.file_location_component import FileLocationComponent

from dam_app.functions.report import get_duplicates_report


@pytest.mark.asyncio
async def test_get_duplicates_report(test_world_alpha: World):
    """Test that the get_duplicates_report function returns the correct duplicate files."""
    async with test_world_alpha.get_context(WorldTransaction)() as transaction:
        session = transaction.session
        # Create an entity with two locations (a duplicate)
        entity1 = await transaction.create_entity()
        hash1 = hashlib.sha256(b"hash1").digest()
        await transaction.add_component_to_entity(entity1.id, ContentHashSHA256Component(hash_value=hash1))
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file1", last_modified_at=datetime.now())
        )
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file2", last_modified_at=datetime.now())
        )
        # Create a second entity with only one location (not a duplicate)
        entity2 = await transaction.create_entity()
        hash2 = hashlib.sha256(b"hash2").digest()
        await transaction.add_component_to_entity(entity2.id, ContentHashSHA256Component(hash_value=hash2))
        await transaction.add_component_to_entity(
            entity2.id, FileLocationComponent(url="file3", last_modified_at=datetime.now())
        )

        duplicates = await get_duplicates_report(session)
        assert len(duplicates) == 1
        assert duplicates[0].entity_id == entity1.id
        assert duplicates[0].total_locations == 2
        assert duplicates[0].hash_value == hash1
