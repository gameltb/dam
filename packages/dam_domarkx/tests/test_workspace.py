import pytest
from dam.core.database import DatabaseManager
from dam_test_utils.types import WorldFactory
from domarkx.managers.workspace_manager import WorkspaceManager
from sqlalchemy.future import select

from dam_domarkx import DomarkxPlugin
from dam_domarkx.models.config import DomarkxSettingsComponent
from dam_domarkx.models.git import Branch, Commit


@pytest.mark.asyncio
async def test_create_workspace(world_factory: WorldFactory):
    """Tests that a workspace can be created."""
    world = await world_factory("test_world", [DomarkxSettingsComponent()])
    world.add_plugin(DomarkxPlugin())
    db = world.get_resource(DatabaseManager)

    manager = WorkspaceManager(world)
    workspace = await manager.create_workspace("test_workspace")

    assert workspace is not None
    assert workspace.name == "test_workspace"

    # Verify that the initial commit and branch were created
    async with db.get_db_session() as session:
        result = await session.execute(select(Commit).where(Commit.workspace_id == workspace.workspace_id))
        commits = result.scalars().all()
        assert len(commits) == 1

        result = await session.execute(
            select(Branch).where(Branch.workspace_id == workspace.workspace_id, Branch.name == "main")
        )
        branch = result.scalars().first()
        assert branch is not None
        assert branch.commit_id == commits[0].commit_id
