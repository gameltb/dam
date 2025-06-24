from dataclasses import dataclass

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


# No need for @Base.mapped_as_dataclass
# kw_only=True should now be inherited from Base
@dataclass
class ContentHashComponent(BaseComponent):  # BaseComponent already inherits from Base
    """
    Stores content-based hashes for an entity (e.g., SHA256, MD5).
    These hashes are typically used for identifying duplicate file content.
    """

    __tablename__ = "content_hashes"
    # __mapper_args__ might be needed if there are specific inheritance settings
    # For simple single table inheritance from an abstract base, it's often not needed.

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    hash_type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., "sha256", "md5"
    hash_value: Mapped[str] = mapped_column(String(256), index=True, nullable=False)  # Length depends on hash type

    # Define relationships or back_populates if needed, e.g., to Entity
    # entity: Mapped["Entity"] = relationship(back_populates="content_hashes")
    # This assumes 'content_hashes' is defined on the Entity model.
    # Since BaseComponent already defines a generic 'entity' relationship,
    # we might not need to redefine it here unless we want a specific back_populates.

    __table_args__ = (UniqueConstraint("entity_id", "hash_type", name="uq_content_hash_entity_type"),)

    def __repr__(self):
        return (
            f"ContentHashComponent(id={self.id}, entity_id={self.entity_id}, "
            f"hash_type='{self.hash_type}', hash_value='{self.hash_value[:10]}...')"
        )
