"""Events for the dam_domarkx package."""

import uuid
from dataclasses import dataclass

from dam.events import BaseEvent


@dataclass
class WorkspaceModified(BaseEvent):
    """An event dispatched when a workspace is modified."""

    workspace_id: uuid.UUID
