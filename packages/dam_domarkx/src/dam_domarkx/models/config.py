"""Configuration for the dam_domarkx package."""

from dataclasses import dataclass, field

from dam.models.config import ConfigComponent
from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column


@dataclass
class DomarkxSettingsComponent(ConfigComponent, MappedAsDataclass):
    """The Domarkx settings component."""

    __tablename__ = "domarkx_settings"
    plugin_name: Mapped[str] = field(default_factory=lambda: mapped_column(default="dam-domarkx"))
