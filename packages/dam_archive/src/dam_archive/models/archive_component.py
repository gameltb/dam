"""A component for storing archive information."""

from dam.models.core import BaseComponent
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveComponent(BaseComponent):
    """A component for storing archive information."""

    __tablename__ = "archive"

    format: Mapped[str] = mapped_column(String, nullable=False)
