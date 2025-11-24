from dam.core.database import DatabaseManager
from dam.core.systems import system
from dam.models.core.entity import Entity

from dam_domarkx.commands import CreateWorkspace
from dam_domarkx.models.domarkx import Workspace
from dam_domarkx.models.git import Branch, Commit
from dam_domarkx.utils.hashing import hash_workspace


@system(on_command=CreateWorkspace)
async def create_workspace(cmd: CreateWorkspace, db: DatabaseManager) -> Entity:
    """Create a new workspace."""
    async with db.get_db_session() as session:
        workspace_entity = Entity()
        workspace = Workspace(name=cmd.name)
        workspace.entity = workspace_entity
        session.add(workspace)
        await session.flush()

        commit_entity = Entity()
        initial_commit = Commit(
            workspace_id=workspace.workspace_id,
            parent_id=None,
            hash=await hash_workspace(workspace, db),
        )
        initial_commit.entity = commit_entity
        session.add(initial_commit)
        await session.flush()

        branch_entity = Entity()
        main_branch = Branch(
            workspace_id=workspace.workspace_id,
            name="main",
            commit_id=initial_commit.commit_id,
        )
        main_branch.entity = branch_entity
        session.add(main_branch)
        await session.commit()

    return workspace_entity
