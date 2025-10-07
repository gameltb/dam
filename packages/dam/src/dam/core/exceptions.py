"""Custom exceptions for the DAM ECS framework."""


class DamECSError(Exception):
    """Base class for exceptions raised by the DAM ECS framework."""


class StageExecutionError(DamECSError):
    """Raised when a system within a stage fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        stage_name: str,
        system_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.stage_name = stage_name
        self.system_name = system_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base_str = super().__str__()
        details = f"Stage: {self.stage_name}"
        if self.system_name:
            details += f", Failing System: {self.system_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


class EventHandlingError(DamECSError):
    """Raised when an event handler fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        event_type: str,
        handler_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.event_type = event_type
        self.handler_name = handler_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base_str = super().__str__()
        details = f"Event Type: {self.event_type}"
        if self.handler_name:
            details += f", Failing Handler: {self.handler_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


class CommandHandlingError(DamECSError):
    """Raised when a command handler fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        command_type: str,
        handler_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.command_type = command_type
        self.handler_name = handler_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base_str = super().__str__()
        details = f"Command Type: {self.command_type}"
        if self.handler_name:
            details += f", Failing Handler: {self.handler_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


class EntityNotFoundError(DamECSError):
    """Raised when an entity is not found."""
