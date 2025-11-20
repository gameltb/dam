"""Base class for execution engines."""

import abc


class ExecutionEngine(abc.ABC):
    """Abstract base class for execution engines."""

    @abc.abstractmethod
    def execute(self, code: str) -> str:
        """Execute a block of code."""
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown the execution engine."""
        raise NotImplementedError
