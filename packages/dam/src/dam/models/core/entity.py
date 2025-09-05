from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column

from .base_class import Base

if TYPE_CHECKING:
    pass


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, init=False)

    def __repr__(self) -> str:
        return f"Entity(id={self.id})"
