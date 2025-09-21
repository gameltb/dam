from dataclasses import dataclass
from typing import Optional

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

    message: Optional[str] = None


@dataclass
class ProgressError(SystemProgressEvent):
    """Indicates that an error occurred during the process."""

    exception: Exception
    message: Optional[str] = None


@dataclass
class ProgressUpdate(SystemProgressEvent):
    """Represents a progress update."""

    total: Optional[int] = None
    current: Optional[int] = None
    message: Optional[str] = None
