"""Defines the FilenameComponent model."""

from datetime import datetime

from dam.models.core.base_component import BaseComponent
from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class FilenameComponent(BaseComponent):
    """
    Stores the original filename of an asset and the earliest known time.

    This component helps track the provenance of a filename. The
    `first_seen_at` timestamp is updated only if a newly observed file
    has an earlier timestamp.
    """

    __tablename__ = "component_filename"

    # id, entity_id are inherited from BaseComponent

    filename: Mapped[str | None] = mapped_column(String(), nullable=True, default=None)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    __table_args__ = (
        # An entity should only have one FilenameComponent.
        UniqueConstraint("entity_id", name="uq_filename_entity_id"),
    )

