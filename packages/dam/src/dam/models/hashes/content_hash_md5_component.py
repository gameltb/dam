"""Data model for MD5 content hashes."""

from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import UniqueComponent


class ContentHashMD5Component(UniqueComponent):
    """Stores MD5 content-based hashes (16 bytes) for an entity."""

    __tablename__ = "component_content_hash_md5"

    # MD5 hash is 16 bytes (128 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(16), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_md5_entity_hash"),
        CheckConstraint("length(hash_value) = 16", name="cc_content_hash_md5_hash_value_length"),
    )

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"ContentHashMD5Component(entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
