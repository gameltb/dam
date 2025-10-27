"""Data model for BLAKE3 content hashes."""

from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import UniqueComponent


class ContentHashBLAKE3Component(UniqueComponent):
    """Stores BLAKE3 content-based hashes (32 bytes) for an entity."""

    __tablename__ = "component_content_hash_blake3"

    hash_value: Mapped[bytes] = mapped_column(LargeBinary(32), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_blake3_entity_hash"),
        CheckConstraint("length(hash_value) = 32", name="cc_content_hash_blake3_hash_value_length"),
    )

