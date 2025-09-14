from dataclasses import dataclass

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent
from dam.models.core.component_mixins import UniqueComponentMixin


@dataclass
class MimeTypeComponent(UniqueComponentMixin, BaseComponent):
    __tablename__ = "component_mime_type"
    """
    A component that stores the mime type of an asset.
    """

    value: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
