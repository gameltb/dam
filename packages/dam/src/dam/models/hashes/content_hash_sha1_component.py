from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
)
from sqlalchemy.orm import Mapped, mapped_column, declared_attr

from ..core.base_component import BaseComponent
from ..core.component_mixins import UniqueComponentMixin


class ContentHashSHA1Component(UniqueComponentMixin, BaseComponent):
    """
    Stores SHA1 content-based hashes (20 bytes) for an entity.
    """

    __tablename__ = "component_content_hash_sha1"

    # SHA1 hash is 20 bytes (160 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(20), index=True, nullable=False)

    @declared_attr.directive
    def __table_args__(cls):
        mixin_args = UniqueComponentMixin.__table_args__(cls)
        local_args = (
            CheckConstraint("length(hash_value) = 20", name="cc_content_hash_sha1_hash_value_length"),
        )
        return mixin_args + local_args

    def __repr__(self) -> str:
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return (
            f"ContentHashSHA1Component(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
        )
