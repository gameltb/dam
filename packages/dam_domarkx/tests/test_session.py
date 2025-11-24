import pytest
from dam.core.database import DatabaseManager
from dam.models.core.entity import Entity
from dam.system_events.base import SystemResultEvent
from dam_test_utils.types import WorldFactory
from domarkx.managers.workspace_manager import WorkspaceManager
from sqlalchemy.future import select

from dam_domarkx import DomarkxPlugin
from dam_domarkx.commands import CreateSession, ForkSession
from dam_domarkx.models.config import DomarkxSettingsComponent
from dam_domarkx.models.domarkx import Session
from dam_domarkx.utils.hashing import hash_workspace


@pytest.mark.asyncio
async def test_create_session(world_factory: WorldFactory):
    """Tests that a session can be created."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    create_session_cmd = CreateSession(workspace_id=workspace.workspace_id)
    new_session_entity = None
    async for event in world.dispatch_command(create_session_cmd):
        if isinstance(event, SystemResultEvent):
            new_session_entity = event.result
            break

    async with db.get_db_session() as session:
        result = await session.execute(select(Session).where(Session.entity_id == new_session_entity.id))
        new_session = result.scalars().first()

        assert new_session is not None
        assert new_session.workspace_id == workspace.workspace_id


@pytest.mark.asyncio
async def test_fork_session(world_factory: WorldFactory):
    """Tests that a session can be forked."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    async with db.get_db_session() as session:
        parent_session = Session(workspace_id=workspace.workspace_id, parent_id=None)
        parent_entity = Entity()
        parent_session.entity = parent_entity
        session.add(parent_session)
        await session.commit()

        fork_session_cmd = ForkSession(session_id=parent_session.session_id)
        new_session_entity = None
        async for event in world.dispatch_command(fork_session_cmd):
            if isinstance(event, SystemResultEvent):
                new_session_entity = event.result
                break

        result = await session.execute(select(Session).where(Session.entity_id == new_session_entity.id))
        new_session = result.scalars().first()

        assert new_session is not None
        assert new_session.parent_id == parent_session.session_id


@pytest.mark.asyncio
async def test_hash_workspace(world_factory: WorldFactory):
    """Tests that the workspace hashing is stable."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    hash1 = await hash_workspace(workspace, db)
    hash2 = await hash_workspace(workspace, db)

    assert hash1 == hash2
