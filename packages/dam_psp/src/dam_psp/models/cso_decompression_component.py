"""A component for storing CSO decompression information."""

from dam.models.core import BaseComponent
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class CsoDecompressionComponent(BaseComponent):
    """A component for storing CSO decompression information."""

    __tablename__ = "cso_decompression"

    format: Mapped[str] = mapped_column(String, nullable=False)
