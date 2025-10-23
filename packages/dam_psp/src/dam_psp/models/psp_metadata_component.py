"""A component for storing PSP metadata."""

from typing import Any

from dam.models.core.base_component import BaseComponent, UniqueComponent
from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class PspMetadataComponent(BaseComponent):
    """A component for storing PSP metadata."""

    __tablename__ = "psp_metadata"

    title: Mapped[str] = mapped_column(String, nullable=False)


class PSPSFOMetadataComponent(UniqueComponent):
    """Stores metadata from a PSP ISO's PARAM.SFO file."""

    __tablename__ = "component_psp_sfo_metadata"

    # Common SFO fields
    app_ver: Mapped[str | None] = mapped_column(String())
    bootable: Mapped[int | None] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String())
    disc_id: Mapped[str | None] = mapped_column(String(), index=True)
    disc_version: Mapped[str | None] = mapped_column(String())
    parental_level: Mapped[int | None] = mapped_column(Integer)
    psp_system_ver: Mapped[str | None] = mapped_column(String())
    title: Mapped[str | None] = mapped_column(String(), index=True)

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return f"PSPSFOMetadataComponent(entity_id={self.entity_id}, title='{self.title}', disc_id='{self.disc_id}')"


class PspSfoRawMetadataComponent(UniqueComponent):
    """Stores raw metadata from a PSP ISO's PARAM.SFO file as a JSON object."""

    __tablename__ = "component_psp_sfo_raw_metadata"

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB)

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return (
            f"PspSfoRawMetadataComponent(entity_id={self.entity_id}, "
            f"metadata_json_keys='{list(self.metadata_json.keys())}')"
        )
