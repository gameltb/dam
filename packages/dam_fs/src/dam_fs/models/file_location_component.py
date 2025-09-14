from datetime import datetime

from dam.models.core import BaseComponent
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column


class FileLocationComponent(BaseComponent):
    """
    Stores the physical location or reference of an asset's content.
    An entity's content can exist in multiple locations or be referenced multiple times.
    """

    __tablename__ = "component_file_location"

    # id, entity_id are inherited from BaseComponent

    # The URL representing the file location, e.g., using file:// protocol.
    url: Mapped[str] = mapped_column(String(4096), nullable=False, unique=True)
    last_modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return f"FileLocationComponent(id={self.id}, entity_id={self.entity_id}, url='{self.url}')"
