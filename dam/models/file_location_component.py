from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent

# BaseComponent already brings Base via inheritance.


# kw_only=True and @dataclass behavior are inherited from Base
class FileLocationComponent(BaseComponent):  # BaseComponent inherits from Base(MappedAsDataclass)
    """
    Stores the location of an asset's file.
    An entity can have multiple file locations.
    """

    __tablename__ = "component_file_location"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    file_identifier: Mapped[str] = mapped_column(
        String(256), nullable=False
    )  # SHA256 hash for content-addressable storage
    storage_type: Mapped[str] = mapped_column(
        String(64), default="local_content_addressable", nullable=False
    )  # e.g., "local_content_addressable", "s3"
    original_filename: Mapped[str | None] = mapped_column(
        String(1024), nullable=True
    )  # Optional: original filename for context

    __table_args__ = (
        UniqueConstraint("entity_id", "file_identifier", name="uq_file_location_entity_identifier"),
        # An entity should generally not have the same file content (identifier) listed twice.
    )

    def __repr__(self):
        return (
            f"FileLocationComponent(id={self.id}, entity_id={self.entity_id}, "
            f"type='{self.storage_type}', identifier='{self.file_identifier}', "
            f"original_filename='{self.original_filename}')"
        )
