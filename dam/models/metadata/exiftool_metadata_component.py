"""
This module defines the ExiftoolMetadataComponent for storing raw EXIF data.
"""
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent
from dam.models.core.types import PkId


class ExiftoolMetadataComponent(BaseComponent):
    """
    Component to store raw metadata extracted by exiftool.
    """

    __tablename__ = "exiftool_metadata"

    # Inherits id, entity_id, created_at, updated_at from BaseComponent

    # entity_id: Mapped[PkId] = mapped_column(ForeignKey("entities.id"), index=True, nullable=False, init=False)
    # The entity_id is already defined in BaseComponent and will be inherited.

    raw_exif_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<ExiftoolMetadataComponent id={self.id} entity_id={self.entity_id} data_keys_count='{len(self.raw_exif_json.keys()) if self.raw_exif_json else 0}'>"
