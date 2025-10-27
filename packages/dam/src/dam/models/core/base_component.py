"""Base classes for data components in the DAM system."""

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from .base_class import Base

if TYPE_CHECKING:
    from .entity import Entity


# List to hold all registered component types that inherit from BaseComponent.
# This list will be populated automatically by BaseComponent.__init_subclass__
REGISTERED_COMPONENT_TYPES: list[type["Component"]] = []

# Initialize logger for this module
logger = logging.getLogger(__name__)


class Component(Base):
    """A common, abstract base for all component types."""

    __abstract__ = True

    # This will be defined in subclasses
    entity_id: Mapped[int]

    @declared_attr
    def entity(self) -> Mapped["Entity"]:
        """Relationship to the parent (owning) Entity."""
        return relationship(
            "Entity",
            foreign_keys=[self.entity_id],  # type: ignore
            repr=False,
            init=False,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register concrete component subclasses."""
        super().__init_subclass__(**kwargs)
        is_abstract = cls.__dict__.get("__abstract__", False)
        if not is_abstract:
            if cls not in REGISTERED_COMPONENT_TYPES:
                REGISTERED_COMPONENT_TYPES.append(cls)
                logger.debug("Registered component: %s", cls.__name__)
        else:
            logger.debug("Not registering abstract class: %s", cls.__name__)


class BaseComponent(Component):
    """
    Abstract base class for all components.

    Provides common fields like id, entity_id (linking to an Entity).
    """

    __abstract__ = True

    # Using declared_attr for fields that might need table-specific context.
    # For entity_id, declared_attr ensures it's correctly set up for each subclass's table.
    entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("entities.id"), index=True, nullable=False, init=False
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, init=False)


class UniqueComponent(Component):
    """
    Abstract base class for components that are unique to an entity.

    The entity_id serves as the primary key.
    """

    __abstract__ = True

    entity_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("entities.id"), primary_key=True, init=False)

