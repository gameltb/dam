from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class OriginalSourceInfoComponent(BaseComponent):
    """
    Stores information about an original source file that was processed
    to create or link to an entity's content. An entity can have multiple
    original sources if the same content was ingested from different files.
    """

    __tablename__ = "component_original_source_info"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    original_filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)  # Original full path, if available

    # Timestamp of when this specific source was processed/ingested
    # BaseComponent.created_at can serve this purpose if this component is added upon ingestion.
    # If a separate ingestion_timestamp specific to this source record is needed, it can be added:
    # ingestion_timestamp: Mapped[datetime] = mapped_column(
    #     DateTime(timezone=True), server_default=func.now(), nullable=False
    # )

    # No specific unique constraints here by default, allowing multiple records
    # if the same file (by name/path) is processed multiple times leading to the same entity.
    # If desired, a UniqueConstraint on (entity_id, original_filename, original_path) could be added.

    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of the source (e.g., 'local_file', 'referenced_file', 'web_source').",
    )

    def __repr__(self):
        return (
            f"OriginalSourceInfoComponent(id={self.id}, entity_id={self.entity_id}, "
            f"source_type='{self.source_type}', original_filename='{self.original_filename}', "
            f"original_path='{self.original_path}')"
        )
