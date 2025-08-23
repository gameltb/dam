from sqlalchemy import LargeBinary, UniqueConstraint  # Changed String to LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent


class ImagePerceptualPHashComponent(BaseComponent):
    """
    Stores pHash perceptual hashes (typically 8 bytes for 64-bit) for an image entity.
    """

    __tablename__ = "component_image_perceptual_phash"  # Renamed

    # Perceptual hashes (e.g., pHash from imagehash library) are often 64-bit, so 8 bytes.
    # String(256) was likely oversized for hex representation; 16 chars for 8 bytes hex.
    hash_value: Mapped[bytes] = mapped_column(LargeBinary(8), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_image_phash_entity_hash"),)

    def __repr__(self):
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"ImagePerceptualPHashComponent(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash}')"  # Show full hex for short hashes
