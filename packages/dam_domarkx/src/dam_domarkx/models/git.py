import uuid
from dataclasses import dataclass, field
from typing import Optional

from dam.models.core.base_class import MappedAsDataclass
from dam.models.core.base_component import BaseComponent
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .domarkx import Workspace


@dataclass(kw_only=True)
class Commit(BaseComponent, MappedAsDataclass):
    """Represents a snapshot of a workspace or session."""
    __tablename__ = "component_domarkx_commits"

    commit_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("component_domarkx_commits.commit_id"), nullable=True)
    hash: Mapped[str]


@dataclass(kw_only=True)
class Branch(BaseComponent, MappedAsDataclass):
    """Represents a named branch pointing to a commit."""
    __tablename__ = "component_domarkx_branches"

    branch_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    name: Mapped[str]
    commit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_commits.commit_id"))


@dataclass(kw_only=True)
class Tag(BaseComponent, MappedAsDataclass):
    """Represents a named tag pointing to a commit."""
    __tablename__ = "component_domarkx_tags"

    tag_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default_factory=uuid.uuid4, unique=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_workspaces.workspace_id"))
    name: Mapped[str]
    commit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("component_domarkx_commits.commit_id"))
