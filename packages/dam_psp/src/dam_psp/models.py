"""Defines the database models for PSP-related components."""

from typing import Any

from dam.models.core.base_component import BaseComponent, UniqueComponent
from sqlalchemy import BigInteger, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.orm import Mapped, mapped_column


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


class CsoParentIsoComponent(BaseComponent):
    """Links a virtual ISO entity back to the original CSO file entity from which it was derived."""

    __tablename__ = "component_cso_parent_iso"

    cso_entity_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("entities.id"),
        index=True,
        unique=True,
        nullable=False,
    )

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return f"CsoParentIsoComponent(entity_id={self.entity_id}, cso_entity_id={self.cso_entity_id})"


class IngestedCsoComponent(UniqueComponent):
    """
    A marker component indicating that a CSO file has been successfully ingested.

    This also implies that a corresponding virtual ISO entity has been created.
    """

    __tablename__ = "component_ingested_cso"

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return f"IngestedCsoComponent(entity_id={self.entity_id})"
