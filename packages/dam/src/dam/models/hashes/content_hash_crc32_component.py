"""Data model for CRC32 content hashes."""

from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import UniqueComponent


class ContentHashCRC32Component(UniqueComponent):
    """Stores CRC32 content-based hashes (4 bytes) for an entity."""

    __tablename__ = "component_content_hash_crc32"

    # CRC32 hash is 4 bytes (32 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(4), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_crc32_entity_hash"),
        CheckConstraint("length(hash_value) = 4", name="cc_content_hash_crc32_hash_value_length"),
    )
