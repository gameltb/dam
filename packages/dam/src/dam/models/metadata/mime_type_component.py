from dataclasses import dataclass

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent


@dataclass
class MimeTypeComponent(BaseComponent):
    __tablename__ = "component_mime_type"
    """
    A component that stores the mime type of an asset.
    """

    value: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
