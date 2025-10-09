"""Defines the base class for image perceptual hash components."""

from abc import ABCMeta

from dam.models.core.base_component import BaseComponent
from sqlalchemy import LargeBinary
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm.decl_api import DCTransformDeclarative


class DeclarativeABCMeta(DCTransformDeclarative, ABCMeta):
    """A metaclass that combines SQLAlchemy's DCTransformDeclarative and ABCMeta."""


class BaseImagePerceptualHashComponent(BaseComponent, metaclass=DeclarativeABCMeta):
    """
    Abstract base class for perceptual hash components.

    This class should be inherited by other hash components, but not instantiated directly.
    """

    __abstract__ = True

    hash_value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        hex_hash = self.hash_value.hex() if isinstance(self.hash_value, bytes) else "N/A"
        return f"{self.__class__.__name__}(id={self.id}, entity_id={self.entity_id}, hash_value(hex)='{hex_hash}')"
