from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import UniqueComponent


class ContentHashSHA256Component(UniqueComponent):
    """Stores SHA256 content-based hashes (32 bytes) for an entity."""

    __tablename__ = "component_content_hash_sha256"

    # SHA256 hash is 32 bytes (256 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(32), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "hash_value", name="uq_sha256_hash_value"
        ),  # Hash values themselves are unique across all components
        CheckConstraint("length(hash_value) = 32", name="cc_content_hash_sha256_hash_value_length"),
    )

    def __repr__(self) -> str:
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"ContentHashSHA256Component(entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
