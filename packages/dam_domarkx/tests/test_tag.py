import pytest
from dam.core.database import DatabaseManager
from dam.system_events.base import SystemResultEvent
from dam_test_utils.types import WorldFactory
from domarkx.managers.workspace_manager import WorkspaceManager
from sqlalchemy.future import select

from dam_domarkx import DomarkxPlugin
from dam_domarkx.commands import CreateTag, GetTags
from dam_domarkx.models.config import DomarkxSettingsComponent
from dam_domarkx.models.git import Commit, Tag


@pytest.mark.asyncio
async def test_create_tag(world_factory: WorldFactory):
    """Tests that a tag can be created."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    async with db.get_db_session() as session:
        result = await session.execute(select(Commit).where(Commit.workspace_id == workspace.workspace_id))
        commit = result.scalars().first()

        create_tag_cmd = CreateTag(workspace_id=workspace.workspace_id, name="test_tag", commit_id=commit.commit_id)
        new_tag_entity = None
        async for event in world.dispatch_command(create_tag_cmd):
            if isinstance(event, SystemResultEvent):
                new_tag_entity = event.result
                break

        result = await session.execute(select(Tag).where(Tag.entity_id == new_tag_entity.id))
        new_tag = result.scalars().first()

        assert new_tag is not None
        assert new_tag.name == "test_tag"


@pytest.mark.asyncio
async def test_get_tags(world_factory: WorldFactory):
    """Tests that tags can be retrieved."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    async with db.get_db_session() as session:
        result = await session.execute(select(Commit).where(Commit.workspace_id == workspace.workspace_id))
        commit = result.scalars().first()

        create_tag_cmd = CreateTag(workspace_id=workspace.workspace_id, name="test_tag", commit_id=commit.commit_id)
        async for _ in world.dispatch_command(create_tag_cmd):
            pass

        get_tags_cmd = GetTags(workspace_id=workspace.workspace_id)
        tags = None
        async for event in world.dispatch_command(get_tags_cmd):
            if isinstance(event, SystemResultEvent):
                tags = event.result
                break

        assert len(tags) == 1
        assert tags[0].name == "test_tag"
