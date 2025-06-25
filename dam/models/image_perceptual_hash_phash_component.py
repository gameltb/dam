from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class ImagePerceptualPHashComponent(BaseComponent):
    """
    Stores pHash perceptual hashes for an image entity.
    """

    __tablename__ = "component_image_perceptual_hash_phash"

    hash_value: Mapped[str] = mapped_column(String(256), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_image_phash_entity_hash"),)

    def __repr__(self):
        return (
            f"ImagePerceptualPHashComponent(id={self.id}, entity_id={self.entity_id}, "
            f"hash_value='{self.hash_value[:10]}...')"
        )
