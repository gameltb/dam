"""A plugin that provides a Docker-based sandbox for executing commands."""

import uuid
from typing import Any

import docker
from docker.models.containers import Container

from domarkx.data.models import Resource
from domarkx.plugins.base import Plugin


class DockerSandboxPlugin(Plugin):
    """A plugin that manages Docker containers as sandboxed execution environments."""

    def __init__(self) -> None:
        """Initialize the DockerSandboxPlugin."""
        self._client = docker.from_env()
        self._containers: dict[str, Container] = {}

    @property
    def type(self) -> str:
        """The type of the resource this plugin manages."""
        return "docker_sandbox"

    def create_resource(self, config: dict[str, Any]) -> Resource:
        """
        Create a new Docker container resource.

        The config must contain an 'image' key specifying the Docker image to use.
        """
        if "image" not in config:
            raise ValueError("Docker resource config must contain an 'image' key.")

        image = config["image"]
        resource_id = str(uuid.uuid4())
        container = self._client.containers.run(image, detach=True, tty=True)
        self._containers[resource_id] = container

        return Resource(resource_id=resource_id, type=self.type, config=config)

    def execute_tool(self, resource_id: str, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool in the specified container."""
        if tool_name != "run_command":
            raise ValueError(f"Unsupported tool: {tool_name}")
        if "command" not in kwargs:
            raise ValueError("run_command tool requires a 'command' argument.")

        container = self._containers.get(resource_id)
        if not container:
            raise ValueError(f"Container with ID '{resource_id}' not found.")

        exit_code, output = container.exec_run(kwargs["command"])  # type: ignore
        return {"exit_code": exit_code, "output": output.decode("utf-8")}

    def get_version(self, resource_id: str) -> str:
        """Get the current image ID of the container."""
        container = self._containers.get(resource_id)
        if not container:
            raise ValueError(f"Container with ID '{resource_id}' not found.")
        return container.image.id  # type: ignore

    def commit_version(self, resource_id: str) -> str:
        """Commit the container's current state to a new image."""
        container = self._containers.get(resource_id)
        if not container:
            raise ValueError(f"Container with ID '{resource_id}' not found.")

        image = container.commit()  # type: ignore
        return image.id  # type: ignore

    def load_version(self, resource_id: str, version_id: str) -> None:
        """
        Load a specific version of a resource.

        This is done by stopping the current container and starting a new one
        from the specified image (version_id).
        """
        container = self._containers.get(resource_id)
        if not container:
            raise ValueError(f"Container with ID '{resource_id}' not found.")

        container.stop()
        container.remove()

        new_container = self._client.containers.run(version_id, detach=True, tty=True)
        self._containers[resource_id] = new_container
