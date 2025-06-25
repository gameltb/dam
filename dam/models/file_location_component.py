
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

    __tablename__ = "file_locations"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)  # Path on disk or a reference
    storage_type: Mapped[str] = mapped_column(String(64), default="local", nullable=False)  # e.g., "local", "s3", "url"
    # uri: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    # Optional full URI if different from filepath

    __table_args__ = (
        UniqueConstraint("entity_id", "filepath", name="uq_file_location_entity_filepath"),
        # Consider if filepath alone should be unique globally, or just per entity.
        # Per entity seems more reasonable.
    )

    def __repr__(self):
        return (
            f"FileLocationComponent(id={self.id}, entity_id={self.entity_id}, "
            f"type='{self.storage_type}', path='{self.filepath}')"
        )
