from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class ImagePerceptualDHashComponent(BaseComponent):
    """
    Stores dHash perceptual hashes for an image entity.
    """

    __tablename__ = "component_image_perceptual_hash_dhash"

    hash_value: Mapped[str] = mapped_column(String(256), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_image_dhash_entity_hash"),)

    def __repr__(self):
        return (
            f"ImagePerceptualDHashComponent(id={self.id}, entity_id={self.entity_id}, "
            f"hash_value='{self.hash_value[:10]}...')"
        )
