from dam.models.core.base_component import BaseComponent
from sqlalchemy import LargeBinary, UniqueConstraint  # Changed String to LargeBinary
from sqlalchemy.orm import Mapped, mapped_column


class ImagePerceptualAHashComponent(BaseComponent):
    """
    Stores aHash perceptual hashes (typically 8 bytes for 64-bit) for an image entity.
    """

    __tablename__ = "component_image_perceptual_ahash"  # Renamed

    hash_value: Mapped[bytes] = mapped_column(LargeBinary(8), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_image_ahash_entity_hash"),)

    def __repr__(self) -> str:
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"ImagePerceptualAHashComponent(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash}')"  # Show full hex for short hashes
