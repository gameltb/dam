"""Manages the lifecycle of workspaces."""

from dam.core.world import World
from dam.core.database import DatabaseManager
from dam_domarkx.commands import CreateWorkspace, CreateTag, GetTags
from dam_domarkx.models.domarkx import Workspace
from dam_domarkx.models.git import Tag
from sqlalchemy.future import select
from dam.system_events.base import SystemResultEvent
from typing import List
import uuid


class WorkspaceManager:
    """A central service for managing workspace lifecycles."""

    def __init__(self, world: World) -> None:
        """Initialize the WorkspaceManager."""
        self._world = world
        self._db = world.get_resource(DatabaseManager)

    async def create_workspace(self, name: str) -> Workspace:
        """
        Create a new workspace.

        Args:
            name (str): The name of the workspace to create.

        Returns:
            Workspace: The newly created workspace.

        """
        async with self._db.get_db_session() as session:
            query = await session.execute(select(Workspace).where(Workspace.name == name))
            if query.scalars().first() is not None:
                raise ValueError(f"Workspace with name '{name}' already exists.")

        workspace_entity = None
        async for event in self._world.dispatch_command(CreateWorkspace(name=name)):
            if isinstance(event, SystemResultEvent):
                workspace_entity = event.result
                break
        if workspace_entity is None:
            raise RuntimeError("Failed to create workspace entity.")
        async with self._db.get_db_session() as session:
            query = await session.execute(select(Workspace).where(Workspace.entity_id == workspace_entity.id))
            return query.scalars().one()

    async def get_workspace(self, name: str) -> Workspace | None:
        """
        Get a workspace by its name.

        Args:
            name (str): The name of the workspace to retrieve.

        Returns:
            Workspace | None: The workspace, or None if it does not exist.

        """
        async with self._db.get_session() as session:
            query = await session.execute(select(Workspace).where(Workspace.name == name))
            return query.scalars().first()

    async def create_tag(self, workspace_id: uuid.UUID, name: str, commit_id: uuid.UUID) -> Tag:
        """
        Create a new tag.

        Args:
            workspace_id (uuid.UUID): The ID of the workspace.
            name (str): The name of the tag.
            commit_id (uuid.UUID): The ID of the commit to tag.

        Returns:
            Tag: The newly created tag.

        """
        tag_entity = await self._world.dispatch_command(CreateTag(workspace_id=workspace_id, name=name, commit_id=commit_id)).get_result()
        async with self._db.get_db_session() as session:
            query = await session.execute(select(Tag).where(Tag.entity_id == tag_entity.id))
            return query.scalars().one()

    async def get_tags(self, workspace_id: uuid.UUID) -> List[Tag]:
        """
        Get all tags for a workspace.

        Args:
            workspace_id (uuid.UUID): The ID of the workspace.

        Returns:
            List[Tag]: A list of tags.

        """
        return await self._world.dispatch_command(GetTags(workspace_id=workspace_id)).get_result()
