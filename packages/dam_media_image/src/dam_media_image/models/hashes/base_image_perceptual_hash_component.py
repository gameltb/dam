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
