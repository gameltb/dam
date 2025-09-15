from dataclasses import dataclass
from typing import Optional

from dam.core.system_events import BaseSystemEvent


@dataclass
class StreamingEvent(BaseSystemEvent):
    """Base class for events yielded by a streaming handler."""

    pass


@dataclass
class StreamStarted(StreamingEvent):
    """Indicates that the streaming process has started."""

    pass


@dataclass
class StreamCompleted(StreamingEvent):
    """Indicates that the streaming process has completed successfully."""

    message: Optional[str] = None


@dataclass
class StreamError(StreamingEvent):
    """Indicates that an error occurred during the streaming process."""

    exception: Exception
    message: Optional[str] = None


@dataclass
class StreamProgress(StreamingEvent):
    """Represents a progress update."""

    total: Optional[int] = None
    current: Optional[int] = None
    message: Optional[str] = None
