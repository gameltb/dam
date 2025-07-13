from sqlalchemy import LargeBinary, UniqueConstraint  # Changed String to LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import BaseComponent  # Corrected import


class ContentHashSHA256Component(BaseComponent):
    """
    Stores SHA256 content-based hashes (32 bytes) for an entity.
    """

    __tablename__ = "component_content_hash_sha256"

    # SHA256 hash is 32 bytes (256 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(32), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("entity_id", name="uq_sha256_entity_id"),  # One SHA256 component per entity
        UniqueConstraint(
            "hash_value", name="uq_sha256_hash_value"
        ),  # Hash values themselves are unique across all components
    )

    def __repr__(self):
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return (
            f"ContentHashSHA256Component(id={self.id}, entity_id={self.entity_id}, "
            f"hash_value(hex)='{hex_hash[:10]}...')"
        )
