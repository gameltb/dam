"""Defines the pHash component for storing image perceptual hashes."""

from sqlalchemy import LargeBinary, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from dam_media_image.models.hashes.base_image_perceptual_hash_component import (
    BaseImagePerceptualHashComponent,
)


class ImagePerceptualPHashComponent(BaseImagePerceptualHashComponent):
    """Stores pHash perceptual hashes (typically 8 bytes for 64-bit) for an image entity."""

    __tablename__ = "component_image_perceptual_phash"

    hash_value: Mapped[bytes] = mapped_column(LargeBinary(8), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_image_phash_entity_hash"),)
