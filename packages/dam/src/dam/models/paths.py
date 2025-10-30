"""Components for representing file paths as a tree."""

from sqlalchemy import BigInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .core.base_component import BaseComponent, UniqueComponent


class PathRoot(UniqueComponent):
    """A component to represent the root of a path tree."""

    __tablename__ = "component_path_root"

    path_type: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (UniqueConstraint("path_type", name="uq_path_root_path_type"),)


class PathNode(BaseComponent):
    """A component to represent a segment of a path."""

    __tablename__ = "component_path_node"

    parent_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    segment: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (UniqueConstraint("parent_id", "segment", name="uq_path_node_parent_id_segment"),)
