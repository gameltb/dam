from typing import Optional

from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent

# Base is inherited via BaseComponent


# kw_only=True and @dataclass behavior are inherited from Base
class FilePropertiesComponent(BaseComponent):
    """
    Stores authoritative properties of an entity's associated file,
    such as its original filename, size in bytes, and MIME type.

    This component is the primary source for these file details.
    The `original_filename` could be the name of an uploaded file,
    a filename derived from a URL, or a user-provided name.

    Typically, an entity linked to a file will have one instance of this
    component. Variations (e.g., for different renditions) might be
    handled by separate entities or a more complex component structure
    if needed in the future. The current unique constraint on `entity_id`
    enforces a one-to-one relationship.
    """

    __tablename__ = "component_file_properties"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    original_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, default=None)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True, default=None
    )  # Task 4.2: Added index
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
