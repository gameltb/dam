from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class FileLocationComponent(BaseComponent):
    """
    Stores the physical location or reference of an asset's content.
    An entity's content can exist in multiple locations or be referenced multiple times.
    """

    __tablename__ = "component_file_location"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    # Content identifier, typically the SHA256 hash of the file's content.
    # This links this location entry to the intrinsic content of the asset.
    content_identifier: Mapped[str] = mapped_column(String(256), nullable=False, index=True)

    # The URL representing the file location.
    # Format: dam://<storage_type>/<path_to_file_or_archive>[#<path_inside_archive>]
    url: Mapped[str] = mapped_column(String(4096), nullable=False)

    # Credentials for accessing the file, if any.
    credentials: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    __table_args__ = (
        UniqueConstraint("entity_id", "url", name="uq_url_entity"),
        # An entity should not have the exact same physical file location/reference registered twice.
    )

    def __repr__(self):
        return (
            f"FileLocationComponent(id={self.id}, entity_id={self.entity_id}, "
            f"content_identifier='{self.content_identifier[:12]}...', url='{self.url}')"
        )
