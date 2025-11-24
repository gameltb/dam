from dataclasses import dataclass
from dam.models.config import ConfigComponent
from dam.models.core.base_class import MappedAsDataclass


@dataclass
class DomarkxSettingsComponent(ConfigComponent, MappedAsDataclass):
    __tablename__ = "domarkx_settings"
    plugin_name: str = "dam-domarkx"
