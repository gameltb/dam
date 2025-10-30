"""Tests for asset helper functions."""

import datetime
from pathlib import Path

import pytest
from dam.core.transaction import WorldTransaction
from dam.functions.paths import get_or_create_path_tree_from_path
from dam_fs.models.file_location_component import FileLocationComponent
from dam_test_utils.types import WorldFactory
from rich.console import Console
from sqlalchemy import select

from dam_app.cli.assets import cleanup_deleted_files_logic


@pytest.mark.asyncio
async def test_cleanup_deleted_files_removes_component_for_deleted_file(world_factory: WorldFactory, tmp_path: Path):
    """Test that _cleanup_deleted_files removes the component for a deleted file."""
    world = await world_factory("test_world", [])
    console = Console()
    file_path = tmp_path / "test_file.txt"
    file_path.touch()

    async with world.get_context(WorldTransaction)() as tx:
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, file_path, "filesystem")
        entity = await tx.create_entity()
        await tx.add_component_to_entity(
            entity.id,
            FileLocationComponent(
                url=file_path.as_uri(),
                last_modified_at=datetime.datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

    file_path.unlink()  # Delete the file

    removed_count = await cleanup_deleted_files_logic(world, None, console)
    assert removed_count == 1

    async with world.get_context(WorldTransaction)() as tx:
        result = await tx.session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == entity.id)
        )
        components = result.scalars().all()
        assert len(components) == 0


@pytest.mark.asyncio
async def test_cleanup_deleted_files_keeps_component_for_existing_file(world_factory: WorldFactory, tmp_path: Path):
    """Test that _cleanup_deleted_files keeps the component for an existing file."""
    world = await world_factory("test_world", [])
    console = Console()
    file_path = tmp_path / "test_file.txt"
    file_path.touch()

    async with world.get_context(WorldTransaction)() as tx:
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, file_path, "filesystem")
        entity = await tx.create_entity()
        await tx.add_component_to_entity(
            entity.id,
            FileLocationComponent(
                url=file_path.as_uri(),
                last_modified_at=datetime.datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )

    removed_count = await cleanup_deleted_files_logic(world, None, console)
    assert removed_count == 0

    async with world.get_context(WorldTransaction)() as tx:
        result = await tx.session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == entity.id)
        )
        components = result.scalars().all()
        assert len(components) == 1


@pytest.mark.asyncio
async def test_cleanup_deleted_files_respects_path_filter(world_factory: WorldFactory, tmp_path: Path):
    """Test that _cleanup_deleted_files respects the path filter."""
    world = await world_factory("test_world", [])
    console = Console()

    # Create a file that will be deleted
    file_to_delete = tmp_path / "delete_me.txt"
    file_to_delete.touch()
    async with world.get_context(WorldTransaction)() as tx:
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, file_to_delete, "filesystem")
        entity1 = await tx.create_entity()
        await tx.add_component_to_entity(
            entity1.id,
            FileLocationComponent(
                url=file_to_delete.as_uri(),
                last_modified_at=datetime.datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )
    file_to_delete.unlink()

    # Create another file in a different directory that will also be deleted
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_file_to_delete = other_dir / "delete_me_too.txt"
    other_file_to_delete.touch()
    async with world.get_context(WorldTransaction)() as tx:
        tree_entity_id, node_id = await get_or_create_path_tree_from_path(tx, other_file_to_delete, "filesystem")
        entity2 = await tx.create_entity()
        await tx.add_component_to_entity(
            entity2.id,
            FileLocationComponent(
                url=other_file_to_delete.as_uri(),
                last_modified_at=datetime.datetime.now(),
                tree_entity_id=tree_entity_id,
                node_id=node_id,
            ),
        )
    other_file_to_delete.unlink()

    # Run cleanup only on the `other` directory
    removed_count = await cleanup_deleted_files_logic(world, other_dir, console)
    assert removed_count == 1

    # Check that only the component for the file in the `other` directory was removed
    async with world.get_context(WorldTransaction)() as tx:
        result1 = await tx.session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == entity1.id)
        )
        components1 = result1.scalars().all()
        assert len(components1) == 1  # This one should still exist

        result2 = await tx.session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == entity2.id)
        )
        components2 = result2.scalars().all()
        assert len(components2) == 0  # This one should be gone
