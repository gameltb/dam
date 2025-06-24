from dataclasses import dataclass, field
from sqlalchemy import String, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional

from .base_component import BaseComponent
# Base is inherited via BaseComponent

@dataclass(kw_only=True)
class FilePropertiesComponent(BaseComponent):
    """
    Stores basic properties of an asset's file, such as original name,
    size, and MIME type.
    Typically, an entity would have one primary set of these properties,
    though variations could exist (e.g., for different versions or renditions).
    For now, assuming one per entity via unique constraint.
    """
    __tablename__ = "file_properties"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    original_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # Could add other fields like 'last_modified_on_disk' etc. if needed.

    __table_args__ = (
        # If an entity should only have one FilePropertiesComponent, make entity_id unique.
        # This simplifies things for now. If multiple are needed (e.g. for different
        # file locations of the same entity having slightly different stored props),
        # this constraint would be removed or made composite with other fields.
        UniqueConstraint("entity_id", name="uq_file_properties_entity_id"),
    )

    def __repr__(self):
        return (
            f"FilePropertiesComponent(id={self.id}, entity_id={self.entity_id}, "
            f"filename='{self.original_filename}', size={self.file_size_bytes}, mime='{self.mime_type}')"
        )
