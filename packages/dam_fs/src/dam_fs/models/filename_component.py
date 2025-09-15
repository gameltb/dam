from datetime import datetime
from typing import Optional

from dam.models.core.base_component import BaseComponent
from pydantic import field_validator
from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class FilenameComponent(BaseComponent):
    """
    Stores the original filename of an asset and the earliest known time
    this filename was associated with the asset.

    This component helps track the provenance of a filename. The
    `first_seen_at` timestamp is updated only if a newly observed file
    has an earlier timestamp.
    """

    __tablename__ = "component_filename"

    # id, entity_id are inherited from BaseComponent

    filename: Mapped[Optional[str]] = mapped_column(String(), nullable=True, default=None)
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    __table_args__ = (
        # An entity should only have one FilenameComponent.
        UniqueConstraint("entity_id", name="uq_filename_entity_id"),
    )

    @field_validator("first_seen_at")
    def truncate_microseconds(cls, v: Optional[datetime]) -> Optional[datetime]:
        if isinstance(v, datetime):
            return v.replace(microsecond=0)
        return v

    def __repr__(self) -> str:
        return (
            f"FilenameComponent(id={self.id}, entity_id={self.entity_id}, "
            f"filename='{self.filename}', first_seen_at='{self.first_seen_at}')"
        )
