from dataclasses import dataclass
from dam.events import BaseEvent
import uuid


@dataclass
class WorkspaceModified(BaseEvent):
    workspace_id: uuid.UUID
