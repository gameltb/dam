import uuid
from dataclasses import dataclass, field
from typing import Any

from dam.models.core.base_component import BaseComponent
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column


@dataclass(kw_only=True)
class Workspace(BaseComponent, MappedAsDataclass):
    """Represents a container for versioned resources."""

    __tablename__ = "component_domarkx_workspaces"

    name: Mapped[str] = field(default_factory=lambda: mapped_column(unique=True))
    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )


@dataclass(kw_only=True)
class Session(BaseComponent, MappedAsDataclass):
    """Represents a single, independent conversation."""

    __tablename__ = "component_domarkx_sessions"

    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    )
    parent_id: Mapped[uuid.UUID | None] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_sessions.session_id"), nullable=True)
    )
    session_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )


@dataclass(kw_only=True)
class Message(BaseComponent, MappedAsDataclass):
    """A message in the conversation."""

    __tablename__ = "component_domarkx_messages"

    role: Mapped[str]
    content: Mapped[list[dict[str, Any]]] = field(default_factory=lambda: mapped_column(JSONB, nullable=False))
    session_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_sessions.session_id"))
    )
    message_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )


@dataclass(kw_only=True)
class Resource(BaseComponent, MappedAsDataclass):
    """A base model for a resource within a workspace."""

    __tablename__ = "component_domarkx_resources"

    type: Mapped[str]
    config: Mapped[dict[str, Any]] = field(default_factory=lambda: mapped_column(JSONB, nullable=False))
    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    )
    resource_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )
