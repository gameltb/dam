"""Tests for report helper functions."""

import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from dam.core.transaction import WorldTransaction
from dam.functions.paths import get_or_create_path_tree_from_path
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_test_utils.types import WorldFactory
from rich.console import Console

from dam_app.cli.report import verify_archive_members


@pytest.mark.asyncio
async def test_verify_archive_members_success(world_factory: WorldFactory, tmp_path: Path):
    """Test that _verify_archive_members returns True when archive members match the database."""
    world = await world_factory("test_world", [])
    console = Console()
    archive_path = tmp_path / "archive.zip"

    # Create a dummy zip file
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("member1.txt", b"content1")
        zf.writestr("member2.txt", b"content2")

    hash1 = hashlib.sha256(b"content1").digest()
    hash2 = hashlib.sha256(b"content2").digest()

    async with world.get_context(WorldTransaction)() as transaction:
        # Create entities for the archive and its members
        archive_entity = await transaction.create_entity()
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(
            transaction, archive_path.resolve(), "filesystem"
        )
        await transaction.add_component_to_entity(
            archive_entity.id,
            FileLocationComponent(
                url=f"file://{archive_path.resolve()}",
                last_modified_at=datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

        member1_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(member1_entity.id, ContentHashSHA256Component(hash_value=hash1))
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(transaction, "member1.txt", "archive")
        await transaction.add_component_to_entity(
            member1_entity.id,
            ArchiveMemberComponent(
                archive_entity_id=archive_entity.id,
                path_in_archive="member1.txt",
                modified_at=datetime.now(),
                compressed_size=None,
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

        member2_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(member2_entity.id, ContentHashSHA256Component(hash_value=hash2))
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(transaction, "member2.txt", "archive")
        await transaction.add_component_to_entity(
            member2_entity.id,
            ArchiveMemberComponent(
                archive_entity_id=archive_entity.id,
                path_in_archive="member2.txt",
                modified_at=datetime.now(),
                compressed_size=None,
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

    result = await verify_archive_members(world, archive_path, console)
    assert result is True


@pytest.mark.asyncio
async def test_verify_archive_members_hash_mismatch(world_factory: WorldFactory, tmp_path: Path):
    """Test that _verify_archive_members returns False when an archive member has a hash mismatch."""
    world = await world_factory("test_world", [])
    console = Console()
    archive_path = tmp_path / "archive.zip"

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("member1.txt", b"content1")

    wrong_hash = hashlib.sha256(b"wrong_content").digest()

    async with world.get_context(WorldTransaction)() as transaction:
        archive_entity = await transaction.create_entity()
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(
            transaction, archive_path.resolve(), "filesystem"
        )
        await transaction.add_component_to_entity(
            archive_entity.id,
            FileLocationComponent(
                url=f"file://{archive_path.resolve()}",
                last_modified_at=datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

        member1_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(member1_entity.id, ContentHashSHA256Component(hash_value=wrong_hash))
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(transaction, "member1.txt", "archive")
        await transaction.add_component_to_entity(
            member1_entity.id,
            ArchiveMemberComponent(
                archive_entity_id=archive_entity.id,
                path_in_archive="member1.txt",
                modified_at=datetime.now(),
                compressed_size=None,
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

    result = await verify_archive_members(world, archive_path, console)
    assert result is False


@pytest.mark.asyncio
async def test_verify_archive_members_file_not_found(world_factory: WorldFactory, tmp_path: Path):
    """Test that _verify_archive_members returns False when a member file is not found in the archive."""
    world = await world_factory("test_world", [])
    console = Console()
    archive_path = tmp_path / "archive.zip"

    # Create an empty zip file
    with zipfile.ZipFile(archive_path, "w"):
        pass

    hash1 = hashlib.sha256(b"content1").digest()

    async with world.get_context(WorldTransaction)() as transaction:
        archive_entity = await transaction.create_entity()
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(
            transaction, archive_path.resolve(), "filesystem"
        )
        await transaction.add_component_to_entity(
            archive_entity.id,
            FileLocationComponent(
                url=f"file://{archive_path.resolve()}",
                last_modified_at=datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

        member1_entity = await transaction.create_entity()
        await transaction.add_component_to_entity(member1_entity.id, ContentHashSHA256Component(hash_value=hash1))
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(transaction, "member1.txt", "archive")
        await transaction.add_component_to_entity(
            member1_entity.id,
            ArchiveMemberComponent(
                archive_entity_id=archive_entity.id,
                path_in_archive="member1.txt",  # This file does not exist in the zip
                modified_at=datetime.now(),
                compressed_size=None,
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

    result = await verify_archive_members(world, archive_path, console)
    assert result is False
