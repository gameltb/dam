from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column  # Removed relationship

from .base_class import Base

# BaseComponent no longer needed for a direct relationship here
# from .base_component import BaseComponent

if TYPE_CHECKING:
    # Still useful if any methods on Entity were to type hint with BaseComponent
    pass


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, init=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        init=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        init=False,
    )

    # components: Mapped[List[BaseComponent]] = relationship(...) # REMOVED ENTIRELY

    def __repr__(self):
        return f"Entity(id={self.id})"
