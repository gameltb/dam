from typing import Any, Dict, Optional

from dam.models.core.base_component import BaseComponent
from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class PSPSFOMetadataComponent(BaseComponent):
    """
    Stores metadata from a PSP ISO's PARAM.SFO file.
    """

    __tablename__ = "component_psp_sfo_metadata"

    # Common SFO fields
    app_ver: Mapped[Optional[str]] = mapped_column(String(255))
    bootable: Mapped[Optional[int]] = mapped_column(Integer)
    category: Mapped[Optional[str]] = mapped_column(String(255))
    disc_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    disc_version: Mapped[Optional[str]] = mapped_column(String(255))
    parental_level: Mapped[Optional[int]] = mapped_column(Integer)
    psp_system_ver: Mapped[Optional[str]] = mapped_column(String(255))
    title: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    def __repr__(self) -> str:
        return (
            f"PSPSFOMetadataComponent(id={self.id}, entity_id={self.entity_id}, "
            f"title='{self.title}', disc_id='{self.disc_id}')"
        )


class PspSfoRawMetadataComponent(BaseComponent):
    """
    Stores raw metadata from a PSP ISO's PARAM.SFO file as a JSON object.
    """

    __tablename__ = "component_psp_sfo_raw_metadata"

    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    def __repr__(self) -> str:
        return (
            f"PspSfoRawMetadataComponent(id={self.id}, entity_id={self.entity_id}, "
            f"metadata_json_keys='{list(self.metadata_json.keys())}')"
        )
