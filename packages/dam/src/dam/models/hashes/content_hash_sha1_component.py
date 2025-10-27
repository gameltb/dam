"""Data model for SHA1 content hashes."""

from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import UniqueComponent


class ContentHashSHA1Component(UniqueComponent):
    """Stores SHA1 content-based hashes (20 bytes) for an entity."""

    __tablename__ = "component_content_hash_sha1"

    # SHA1 hash is 20 bytes (160 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(20), index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_sha1_entity_hash"),
        CheckConstraint("length(hash_value) = 20", name="cc_content_hash_sha1_hash_value_length"),
    )

