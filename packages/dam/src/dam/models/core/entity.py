"""Data model for the core Entity."""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from .base_class import Base

if TYPE_CHECKING:
    pass


class Entity(Base):
    """Represents a unique entity in the system, acting as a container for components."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, init=False)
