import uuid
from dataclasses import dataclass, field
from typing import Optional

from dam.models.core.base_class import MappedAsDataclass
from dam.models.core.base_component import BaseComponent
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


@dataclass(kw_only=True)
class Workspace(BaseComponent, MappedAsDataclass):
    """Represents a container for versioned resources."""
    __tablename__ = "component_domarkx_workspaces"

    workspace_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    name: Mapped[str] = mapped_column(unique=True)


@dataclass(kw_only=True)
class Session(BaseComponent, MappedAsDataclass):
    """Represents a single, independent conversation."""
    __tablename__ = "component_domarkx_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("component_domarkx_sessions.session_id"), nullable=True)


@dataclass(kw_only=True)
class Message(BaseComponent, MappedAsDataclass):
    """A message in the conversation."""
    __tablename__ = "component_domarkx_messages"

    message_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_sessions.session_id"))
    role: Mapped[str]
    content: Mapped[list] = mapped_column(JSONB, nullable=False)


@dataclass(kw_only=True)
class Resource(BaseComponent, MappedAsDataclass):
    """A base model for a resource within a workspace."""
    __tablename__ = "component_domarkx_resources"

    resource_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    type: Mapped[str]
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
