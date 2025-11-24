"""Commands for the dam_domarkx package."""

import uuid
from dataclasses import dataclass

from dam.commands.core import BaseCommand
from dam.models.core.entity import Entity
from dam.system_events.base import BaseSystemEvent

from dam_domarkx.models.git import Tag


@dataclass
class CreateWorkspace(BaseCommand[Entity, BaseSystemEvent]):
    """A command to create a new workspace."""

    name: str


@dataclass
class ForkSession(BaseCommand[Entity, BaseSystemEvent]):
    """A command to fork a session."""

    session_id: uuid.UUID


@dataclass
class CreateSession(BaseCommand[Entity, BaseSystemEvent]):
    """A command to create a new session."""

    workspace_id: uuid.UUID


@dataclass
class CreateTag(BaseCommand[Entity, BaseSystemEvent]):
    """A command to create a new tag."""

    workspace_id: uuid.UUID
    name: str
    commit_id: uuid.UUID


@dataclass
class GetTags(BaseCommand[list[Tag], BaseSystemEvent]):
    """A command to get all tags for a workspace."""

    workspace_id: uuid.UUID
