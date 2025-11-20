"""Manages the lifecycle of workspaces."""

from domarkx.data.models import Workspace


class WorkspaceManager:
    """A central service for managing workspace lifecycles."""

    def __init__(self) -> None:
        """Initialize the WorkspaceManager."""
        self._workspaces: dict[str, Workspace] = {}
        # In the future, this manager will also be responsible for loading plugins.

    def create_workspace(self, workspace_id: str) -> Workspace:
        """
        Create a new workspace.

        Args:
            workspace_id (str): The ID of the workspace to create.

        Returns:
            Workspace: The newly created workspace.

        """
        if workspace_id in self._workspaces:
            raise ValueError(f"Workspace with ID '{workspace_id}' already exists.")
        workspace = Workspace(workspace_id=workspace_id)
        self._workspaces[workspace_id] = workspace
        return workspace

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        """
        Get a workspace by its ID.

        Args:
            workspace_id (str): The ID of the workspace to retrieve.

        Returns:
            Workspace | None: The workspace, or None if it does not exist.

        """
        return self._workspaces.get(workspace_id)
