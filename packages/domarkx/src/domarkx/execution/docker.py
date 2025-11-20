"""Docker execution engine."""

from domarkx.execution.engine import ExecutionEngine


class DockerEngine(ExecutionEngine):
    """An execution engine that runs code in a Docker container."""

    def execute(self, code: str) -> str:
        """Execute a block of code in a Docker container."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the Docker container."""
        raise NotImplementedError
