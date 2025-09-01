from sqlalchemy import LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent


class ContentHashBLAKE3Component(BaseComponent):
    __tablename__ = "component_content_hash_blake3"

    hash_value: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False, unique=True, index=True)

    def __repr__(self) -> str:
        return f"ContentHashBLAKE3Component(id={self.id}, hash_value={self.hash_value.hex()})"
