from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
)
from sqlalchemy.orm import Mapped, mapped_column, declared_attr

from ..core.base_component import BaseComponent
from ..core.component_mixins import UniqueComponentMixin


class ContentHashBLAKE3Component(UniqueComponentMixin, BaseComponent):
    """
    Stores BLAKE3 content-based hashes (32 bytes) for an entity.
    """

    __tablename__ = "component_content_hash_blake3"

    hash_value: Mapped[bytes] = mapped_column(LargeBinary(32), index=True, nullable=False)

    @declared_attr.directive
    def __table_args__(cls):
        mixin_args = UniqueComponentMixin.__table_args__(cls)
        local_args = (
            CheckConstraint("length(hash_value) = 32", name="cc_content_hash_blake3_hash_value_length"),
        )
        return mixin_args + local_args

    def __repr__(self) -> str:
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"ContentHashBLAKE3Component(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
