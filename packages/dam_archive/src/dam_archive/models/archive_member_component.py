"""Defines the ArchiveMemberComponent."""

from datetime import datetime

from dam.models.core import BaseComponent as Component
from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveMemberComponent(Component):
    """A component that marks an asset as a member of an archive."""

    __tablename__ = "component_archive_member"

    archive_entity_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("entities.id"), nullable=False, index=True)
    path_in_archive: Mapped[str] = mapped_column(String(), nullable=False)
    modified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    compressed_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    tree_entity_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("entities.id"), nullable=True)
    node_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("component_path_node.id"), nullable=True)
