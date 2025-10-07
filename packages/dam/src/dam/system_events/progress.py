"""System events for reporting progress."""

from dataclasses import dataclass

from .base import BaseSystemEvent


@dataclass
class SystemProgressEvent(BaseSystemEvent):
    """Base class for events yielded by a streaming handler to report progress."""

    pass


@dataclass
class ProgressStarted(SystemProgressEvent):
    """Indicates that the progress-reporting process has started."""

    pass


@dataclass
class ProgressCompleted(SystemProgressEvent):
    """Indicates that the process has completed successfully."""

    message: str | None = None


@dataclass
class ProgressError(SystemProgressEvent):
    """Indicates that an error occurred during the process."""

    exception: Exception
    message: str | None = None


@dataclass
class ProgressUpdate(SystemProgressEvent):
    """Represents a progress update."""

    total: int | None = None
    current: int | None = None
    message: str | None = None
