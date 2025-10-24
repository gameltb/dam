"""DAM PSP SFO metadata models."""

from dataclasses import dataclass
from typing import Any

from dam.models.core.base_component import BaseComponent
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column


@dataclass
class PSPSFOMetadataComponent(BaseComponent):
    """A component that stores extracted metadata from a PSP SFO file."""

    __tablename__ = "component_psp_sfo_metadata"

    app_ver: Mapped[str | None] = mapped_column(String)  # noqa: RUF009
    bootable: Mapped[int | None] = mapped_column()  # noqa: RUF009
    category: Mapped[str | None] = mapped_column(String)  # noqa: RUF009
    disc_id: Mapped[str | None] = mapped_column(String)  # noqa: RUF009
    disc_version: Mapped[str | None] = mapped_column(String)  # noqa: RUF009
    parental_level: Mapped[int | None] = mapped_column()  # noqa: RUF009
    psp_system_ver: Mapped[str | None] = mapped_column(String)  # noqa: RUF009
    title: Mapped[str | None] = mapped_column(String)  # noqa: RUF009


@dataclass
class PspSfoRawMetadataComponent(BaseComponent):
    """A component that stores raw SFO metadata as a JSON object."""

    __tablename__ = "component_psp_sfo_raw_metadata"

    metadata_json: Mapped[dict[Any, Any] | None] = mapped_column(JSON, default=None)  # noqa: RUF009
