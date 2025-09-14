from sqlalchemy import (
    CheckConstraint,
    LargeBinary,
)
from sqlalchemy.orm import Mapped, mapped_column, declared_attr

from ..core.base_component import BaseComponent
from ..core.component_mixins import UniqueComponentMixin


class ContentHashMD5Component(UniqueComponentMixin, BaseComponent):
    """
    Stores MD5 content-based hashes (16 bytes) for an entity.
    """

    __tablename__ = "component_content_hash_md5"

    # MD5 hash is 16 bytes (128 bits)
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(16), index=True, nullable=False)

    @declared_attr.directive
    def __table_args__(cls):
        mixin_args = UniqueComponentMixin.__table_args__(cls)
        local_args = (
            CheckConstraint("length(hash_value) = 16", name="cc_content_hash_md5_hash_value_length"),
        )
        return mixin_args + local_args

    def __repr__(self) -> str:
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return (
            f"ContentHashMD5Component(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash[:10]}...')"
        )
