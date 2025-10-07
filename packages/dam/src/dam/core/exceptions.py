"""Custom exceptions for the DAM ECS framework."""



class DamECSException(Exception):
    """Base class for exceptions raised by the DAM ECS framework."""

    pass


class StageExecutionError(DamECSException):
    """Raised when a system within a stage fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        stage_name: str,
        system_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stage_name = stage_name
        self.system_name = system_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        base_str = super().__str__()
        details = f"Stage: {self.stage_name}"
        if self.system_name:
            details += f", Failing System: {self.system_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


class EventHandlingError(DamECSException):
    """Raised when an event handler fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        event_type: str,
        handler_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.event_type = event_type
        self.handler_name = handler_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        base_str = super().__str__()
        details = f"Event Type: {self.event_type}"
        if self.handler_name:
            details += f", Failing Handler: {self.handler_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


class ResourceNotFoundError(DamECSException, LookupError):  # Inherit from LookupError for semantic correctness
    """Raised when a requested resource is not found in the ResourceManager."""

    pass


# It seems ResourceNotFoundError was already defined in dam.core.resources
# I should ensure this new definition is compatible or use the existing one.
# For now, I'll assume I'm defining it here for centralization,
# and would remove the one in dam.core.resources.
# For this step, I'll just define StageExecutionError and EventHandlingError.
# I'll check dam.core.resources.py for ResourceNotFoundError later if conflicts arise.
# Let's remove ResourceNotFoundError from here to avoid conflict for now.


class CommandHandlingError(DamECSException):
    """Raised when a command handler fails, leading to a rollback."""

    def __init__(
        self,
        message: str,
        command_type: str,
        handler_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.command_type = command_type
        self.handler_name = handler_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        base_str = super().__str__()
        details = f"Command Type: {self.command_type}"
        if self.handler_name:
            details += f", Failing Handler: {self.handler_name}"
        if self.original_exception:
            details += f", Original Error: {type(self.original_exception).__name__}: {self.original_exception}"
        return f"{base_str} ({details})"


del ResourceNotFoundError  # Removing this to avoid conflict with existing one.
# DamECSException, StageExecutionError, EventHandlingError are the new ones for this task.


class EntityNotFoundError(DamECSException):
    """Raised when an entity is not found."""

    pass
