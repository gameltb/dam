"""A component for storing the content of an asset."""

from sqlalchemy import LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core import BaseComponent


class ContentComponent(BaseComponent):
    """A component for storing the content of an asset."""

    __tablename__ = "content"

    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
