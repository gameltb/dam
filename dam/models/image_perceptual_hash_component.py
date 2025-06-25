from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


# No need for @Base.mapped_as_dataclass
# kw_only=True and @dataclass behavior are inherited from Base
class ImagePerceptualHashComponent(BaseComponent):  # BaseComponent already inherits from Base
    """
    Stores perceptual hashes for an image entity (e.g., pHash, aHash, dHash).
    These hashes are used for finding visually similar images.
    """

    __tablename__ = "image_perceptual_hashes"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    hash_type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., "phash", "ahash", "dhash"
    hash_value: Mapped[str] = mapped_column(
        String(256), index=True, nullable=False
    )  # Perceptual hashes can be hex strings

    # Optional: Store parameters used to generate the hash, if they can vary
    # hash_settings: Mapped[str] = mapped_column(String(256), nullable=True) # e.g., JSON string or specific format

    # As with ContentHashComponent, the 'entity' relationship is inherited from BaseComponent.
    # If a specific back_populates="perceptual_hashes" were needed on Entity, it would be defined here.

    __table_args__ = (UniqueConstraint("entity_id", "hash_type", name="uq_image_perceptual_hash_entity_type"),)

    def __repr__(self):
        return (
            f"ImagePerceptualHashComponent(id={self.id}, entity_id={self.entity_id}, "
            f"hash_type='{self.hash_type}', hash_value='{self.hash_value[:10]}...')"
        )
