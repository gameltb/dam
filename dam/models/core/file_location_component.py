from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .. import BaseComponent


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

    storage_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # e.g., "local_cas" (Content Addressable Storage), "local_reference", "s3_cas"

    # The actual path or key for accessing the file in the given storage_type.
    # For "local_cas": relative path within that CAS store (e.g., "ab/cd/hashvalue")
    # For "local_reference": absolute path to the original file.
    # For "s3_cas": S3 object key.
    physical_path_or_key: Mapped[str] = mapped_column(String(2048), nullable=False)

    # Optional: original filename associated with this specific location instance, if meaningful.
    # Broader tracking of all original filenames that map to this content should be in OriginalSourceInfoComponent.
    contextual_filename: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    __table_args__ = (
        UniqueConstraint("entity_id", "storage_type", "physical_path_or_key", name="uq_physical_location_entity"),
        # An entity should not have the exact same physical file location/reference registered twice.
    )

    def __repr__(self):
        return (
            f"FileLocationComponent(id={self.id}, entity_id={self.entity_id}, "
            f"content_identifier='{self.content_identifier[:12]}...', storage_type='{self.storage_type}', "
            f"physical_path_or_key='{self.physical_path_or_key}', contextual_filename='{self.contextual_filename}')"
        )
