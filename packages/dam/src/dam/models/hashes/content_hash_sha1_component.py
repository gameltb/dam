from sqlalchemy import LargeBinary, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import BaseComponent


class ContentHashSHA1Component(BaseComponent):
    """
    Stores SHA1 content-based hashes (20 bytes) for an entity.
    """

    __tablename__ = "component_content_hash_sha1"

    # SHA1 hash is 20 bytes (160 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(20), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_sha1_entity_hash"),)

    def __repr__(self):
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return (
            f"ContentHashSHA1Component(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
        )
