import uuid
from dataclasses import dataclass, field

from dam.models.core.base_component import BaseComponent
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column


@dataclass(kw_only=True)
class Commit(BaseComponent, MappedAsDataclass):
    """Represents a snapshot of a workspace or session."""

    __tablename__ = "component_domarkx_commits"

    hash: Mapped[str]
    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    )
    parent_id: Mapped[uuid.UUID | None] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_commits.commit_id"), nullable=True)
    )
    commit_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )


@dataclass(kw_only=True)
class Branch(BaseComponent, MappedAsDataclass):
    """Represents a named branch pointing to a commit."""

    __tablename__ = "component_domarkx_branches"

    name: Mapped[str]
    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    )
    commit_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_commits.commit_id"))
    )
    branch_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )


@dataclass(kw_only=True)
class Tag(BaseComponent, MappedAsDataclass):
    """Represents a named tag pointing to a commit."""

    __tablename__ = "component_domarkx_tags"

    name: Mapped[str]
    workspace_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    )
    commit_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(ForeignKey("component_domarkx_commits.commit_id"))
    )
    tag_id: Mapped[uuid.UUID] = field(
        default_factory=lambda: mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    )
