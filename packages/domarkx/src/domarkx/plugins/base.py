"""Base classes for the plugin system."""

import abc
from typing import Any

from domarkx.data.models import Resource


class Plugin(abc.ABC):
    """Abstract base class for a domarkx plugin."""

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """The type of the resource this plugin manages (e.g., 'docker_sandbox')."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_resource(self, config: dict[str, Any]) -> Resource:
        """
        Create a new resource instance.

        Args:
            config (dict): The configuration for the resource.

        Returns:
            Resource: The newly created resource.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_tool(self, resource_id: str, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute a tool on a resource.

        Args:
            resource_id (str): The ID of the resource to execute the tool on.
            tool_name (str): The name of the tool to execute.
            **kwargs: The arguments for the tool.

        Returns:
            Any: The result of the tool execution.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_version(self, resource_id: str) -> str:
        """
        Get the current version of a resource.

        Args:
            resource_id (str): The ID of the resource.

        Returns:
            str: The current version identifier of the resource.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def commit_version(self, resource_id: str) -> str:
        """
        Create a new version of a resource.

        Args:
            resource_id (str): The ID of the resource.

        Returns:
            str: The new version identifier of the resource.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_version(self, resource_id: str, version_id: str) -> None:
        """
        Load a specific version of a resource.

        Args:
            resource_id (str): The ID of the resource.
            version_id (str): The version to load.

        """
        raise NotImplementedError
