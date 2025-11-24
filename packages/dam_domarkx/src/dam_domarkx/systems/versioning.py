from dam.core.database import DatabaseManager
from dam.core.systems import system
from dam.models.core.entity import Entity
from sqlalchemy.future import select

from dam_domarkx.events import WorkspaceModified
from dam_domarkx.models.domarkx import Workspace
from dam_domarkx.models.git import Branch, Commit
from dam_domarkx.utils.hashing import hash_workspace


@system(on_event=WorkspaceModified)
class WorkspaceVersioningSystem:
    """A system that automatically creates commits for modified workspaces."""

    async def run(self, event: WorkspaceModified, db: DatabaseManager):
        """Run the system."""
        async with db.get_db_session() as session:
            result = await session.execute(select(Workspace).where(Workspace.workspace_id == event.workspace_id))
            workspace = result.scalars().first()
            if workspace:
                new_hash = await hash_workspace(workspace, db)

                main_branch = (
                    (
                        await session.execute(
                            select(Branch).where(Branch.workspace_id == workspace.workspace_id, Branch.name == "main")
                        )
                    )
                    .scalars()
                    .first()
                )
                if main_branch:
                    latest_commit = (
                        (await session.execute(select(Commit).where(Commit.commit_id == main_branch.commit_id)))
                        .scalars()
                        .first()
                    )
                    if latest_commit and latest_commit.hash != new_hash:
                        commit_entity = Entity()
                        new_commit = Commit(
                            workspace_id=workspace.workspace_id,
                            parent_id=main_branch.commit_id,
                            hash=new_hash,
                        )
                        new_commit.entity = commit_entity
                        session.add(new_commit)
                        main_branch.commit_id = new_commit.commit_id
            await session.commit()
