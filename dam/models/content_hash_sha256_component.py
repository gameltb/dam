from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class ContentHashSHA256Component(BaseComponent):
    """
    Stores SHA256 content-based hashes for an entity.
    """

    __tablename__ = "component_content_hash_sha256"

    hash_value: Mapped[str] = mapped_column(String(64), index=True, nullable=False)  # SHA256 hashes are 64 chars

    __table_args__ = (
        UniqueConstraint("entity_id", name="uq_sha256_entity_id"),  # One SHA256 component per entity
        UniqueConstraint(
            "hash_value", name="uq_sha256_hash_value"
        ),  # Hash values themselves are unique across all components
    )

    def __repr__(self):
        return (
            f"ContentHashSHA256Component(id={self.id}, entity_id={self.entity_id}, "
            f"hash_value='{self.hash_value[:10]}...')"
        )
