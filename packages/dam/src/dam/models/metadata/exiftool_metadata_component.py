"""
This module defines the ExiftoolMetadataComponent for storing raw EXIF data.
"""

from typing import Any

from sqlalchemy import JSON  # Changed from JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent


class ExiftoolMetadataComponent(BaseComponent):
    """
    Component to store raw metadata extracted by exiftool.
    """

    __tablename__ = "component_exiftool_metadata"  # Renamed

    # Inherits id, entity_id from BaseComponent

    # entity_id: Mapped[PkId] = mapped_column(ForeignKey("entities.id"), index=True, nullable=False, init=False)
    # The entity_id is already defined in BaseComponent and will be inherited.

    raw_exif_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<ExiftoolMetadataComponent id={self.id} entity_id={self.entity_id} data_keys_count='{len(self.raw_exif_json.keys()) if self.raw_exif_json else 0}'>"
