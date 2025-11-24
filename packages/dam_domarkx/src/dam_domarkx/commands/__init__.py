import uuid
from dataclasses import dataclass
from dam.commands.core import BaseCommand


@dataclass
class CreateWorkspace(BaseCommand):
    name: str

@dataclass
class ForkSession(BaseCommand):
    session_id: uuid.UUID

@dataclass
class CreateSession(BaseCommand):
    workspace_id: uuid.UUID

@dataclass
class CreateTag(BaseCommand):
    workspace_id: uuid.UUID
    name: str
    commit_id: uuid.UUID

@dataclass
class GetTags(BaseCommand):
    workspace_id: uuid.UUID
