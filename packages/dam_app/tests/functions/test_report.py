"""Tests for report functions."""

import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from dam.core.transaction import WorldTransaction
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_test_utils.types import WorldFactory

from dam_app.functions.report import create_delete_plan, get_duplicates_report


@pytest.mark.asyncio
async def test_get_duplicates_report(world_factory: WorldFactory):
    """Test that the get_duplicates_report function returns the correct duplicate files."""
    world = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as transaction:
        # Create an entity with two locations (a duplicate)
        entity1 = await transaction.create_entity()
        hash1 = hashlib.sha256(b"hash1").digest()
        await transaction.add_component_to_entity(entity1.id, ContentHashSHA256Component(hash_value=hash1))
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file:///tmp/file1", last_modified_at=datetime.now())
        )
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file:///tmp/file2", last_modified_at=datetime.now())
        )
        # Create a second entity with only one location (not a duplicate)
        entity2 = await transaction.create_entity()
        hash2 = hashlib.sha256(b"hash2").digest()
        await transaction.add_component_to_entity(entity2.id, ContentHashSHA256Component(hash_value=hash2))
        await transaction.add_component_to_entity(
            entity2.id, FileLocationComponent(url="file:///tmp/file3", last_modified_at=datetime.now())
        )

        duplicates = await get_duplicates_report(transaction.session)
        assert len(duplicates) == 2
        assert duplicates[0].entity_id == entity1.id
        assert duplicates[0].total_locations == 2
        assert duplicates[0].hash_value == hash1
        assert duplicates[1].entity_id == entity1.id
        assert duplicates[1].total_locations == 2
        assert duplicates[1].hash_value == hash1


@pytest.mark.asyncio
async def test_get_duplicates_report_with_path_filter(world_factory: WorldFactory):
    """Test that the get_duplicates_report function returns the correct duplicate files when a path filter is applied."""
    world = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as transaction:
        # Create an entity with two locations (a duplicate)
        entity1 = await transaction.create_entity()
        hash1 = hashlib.sha256(b"hash1").digest()
        await transaction.add_component_to_entity(entity1.id, ContentHashSHA256Component(hash_value=hash1))
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file:///tmp/file1", last_modified_at=datetime.now())
        )
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file:///data/file2", last_modified_at=datetime.now())
        )
        # Create a second entity with only one location (not a duplicate)
        entity2 = await transaction.create_entity()
        hash2 = hashlib.sha256(b"hash2").digest()
        await transaction.add_component_to_entity(entity2.id, ContentHashSHA256Component(hash_value=hash2))
        await transaction.add_component_to_entity(
            entity2.id, FileLocationComponent(url="file:///tmp/file3", last_modified_at=datetime.now())
        )

        duplicates = await get_duplicates_report(transaction.session, Path("/tmp"))
        assert len(duplicates) == 1
        assert duplicates[0].entity_id == entity1.id


@pytest.mark.asyncio
async def test_get_duplicates_report_with_indirect_path_filter(world_factory: WorldFactory):
    """Test that the get_duplicates_report function returns the correct duplicate files when an indirect path filter is applied."""
    world = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as transaction:
        # Create an entity that represents the content of the duplicate files
        entity1 = await transaction.create_entity()
        hash1 = hashlib.sha256(b"hash1").digest()
        await transaction.add_component_to_entity(entity1.id, ContentHashSHA256Component(hash_value=hash1))

        # Location 1: Inside an archive at /tmp/archive.zip
        archive_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(
            archive_entity.id, FileLocationComponent(url="file:///tmp/archive.zip", last_modified_at=datetime.now())
        )
        await transaction.add_component_to_entity(
            entity1.id,
            ArchiveMemberComponent(
                archive_entity_id=archive_entity.id,
                path_in_archive="file1",
                modified_at=datetime.now(),
                compressed_size=None,
            ),
        )

        # Location 2: A direct file location not under /tmp
        await transaction.add_component_to_entity(
            entity1.id, FileLocationComponent(url="file:///data/file2", last_modified_at=datetime.now())
        )

        # Create another non-duplicate entity to ensure it's not picked up
        entity2 = await transaction.create_entity()
        hash2 = hashlib.sha256(b"hash2").digest()
        await transaction.add_component_to_entity(entity2.id, ContentHashSHA256Component(hash_value=hash2))
        await transaction.add_component_to_entity(
            entity2.id, FileLocationComponent(url="file:///tmp/file3", last_modified_at=datetime.now())
        )

        # Since entity1 has two locations, it is a duplicate.
        # One of its locations is indirectly under /tmp.
        # So when we filter by /tmp, we should get results for entity1.
        duplicates = await get_duplicates_report(transaction.session, Path("/tmp"))

        # The new query returns one row per path.
        # The filter is on the path.
        # The query will first find all duplicates (total_locations > 1). entity1 is a duplicate.
        # Then it will filter these results to only include paths that start with /tmp.
        # In this case, it should return the row for the archive path.
        assert len(duplicates) == 1
        assert duplicates[0].entity_id == entity1.id
        assert duplicates[0].path == "file:///tmp/archive.zip -> file1"


@pytest.mark.asyncio
async def test_create_delete_plan(world_factory: WorldFactory):
    """Test that the create_delete_plan function returns the correct delete plan."""
    world = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as transaction:
        # Source files
        source_entity1 = await transaction.create_entity()
        hash1 = hashlib.sha256(b"hash1").digest()
        await transaction.add_component_to_entity(source_entity1.id, ContentHashSHA256Component(hash_value=hash1))
        await transaction.add_component_to_entity(
            source_entity1.id,
            FileLocationComponent(url="file:///tmp/source/file1", last_modified_at=datetime.now()),
        )

        source_entity2 = await transaction.create_entity()
        hash2 = hashlib.sha256(b"hash2").digest()
        await transaction.add_component_to_entity(source_entity2.id, ContentHashSHA256Component(hash_value=hash2))
        source_archive_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(
            source_archive_entity.id,
            FileLocationComponent(url="file:///tmp/source/archive.zip", last_modified_at=datetime.now()),
        )
        await transaction.add_component_to_entity(
            source_entity2.id,
            ArchiveMemberComponent(
                archive_entity_id=source_archive_entity.id,
                path_in_archive="member2",
                modified_at=datetime.now(),
                compressed_size=None,
            ),
        )

        # Target files
        # A duplicate of source_entity1
        await transaction.add_component_to_entity(
            source_entity1.id,
            FileLocationComponent(url="file:///tmp/target/file1_dup", last_modified_at=datetime.now()),
        )

        # An archive in the target directory where all members are duplicates of source files
        target_archive_entity = await transaction.create_entity()
        archive_hash = hashlib.sha256(b"archive_hash").digest()
        await transaction.add_component_to_entity(
            target_archive_entity.id, ContentHashSHA256Component(hash_value=archive_hash)
        )
        await transaction.add_component_to_entity(
            target_archive_entity.id,
            FileLocationComponent(url="file:///tmp/target/archive.zip", last_modified_at=datetime.now()),
        )
        await transaction.add_component_to_entity(
            source_entity1.id,
            ArchiveMemberComponent(
                archive_entity_id=target_archive_entity.id,
                path_in_archive="member1",
                modified_at=datetime.now(),
                compressed_size=None,
            ),
        )
        await transaction.add_component_to_entity(
            source_entity2.id,
            ArchiveMemberComponent(
                archive_entity_id=target_archive_entity.id,
                path_in_archive="member2",
                modified_at=datetime.now(),
                compressed_size=None,
            ),
        )

        # A file that is not a duplicate
        non_dup_entity = await transaction.create_entity()
        non_dup_hash = hashlib.sha256(b"non_dup").digest()
        await transaction.add_component_to_entity(
            non_dup_entity.id, ContentHashSHA256Component(hash_value=non_dup_hash)
        )
        await transaction.add_component_to_entity(
            non_dup_entity.id,
            FileLocationComponent(url="file:///tmp/target/unique_file", last_modified_at=datetime.now()),
        )

        delete_plan = await create_delete_plan(transaction.session, Path("/tmp/source"), Path("/tmp/target"))

        assert len(delete_plan) == 2

        plan_map = {p.target_path: p for p in delete_plan}

        # Check for the direct duplicate file
        direct_dup_plan = plan_map.get("/tmp/target/file1_dup")
        assert direct_dup_plan is not None
        assert direct_dup_plan.source_path == "/tmp/source/file1"
        assert direct_dup_plan.hash == hash1.hex()
        assert direct_dup_plan.details == "Duplicate of /tmp/source/file1"

        # Check for the archive file to be deleted
        archive_dup_plan = plan_map.get("/tmp/target/archive.zip")
        assert archive_dup_plan is not None
        assert archive_dup_plan.hash == archive_hash.hex()
        assert "'/tmp/target/archive.zip -> member1' is a duplicate of '/tmp/source/file1'" in archive_dup_plan.details
        assert (
            "'/tmp/target/archive.zip -> member2' is a duplicate of '/tmp/source/archive.zip -> member2'"
            in archive_dup_plan.details
        )
