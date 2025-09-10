import logging
from typing import TYPE_CHECKING, Any, List, Type

from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from .base_class import Base

if TYPE_CHECKING:
    from .entity import Entity


# List to hold all registered component types that inherit from BaseComponent.
# This list will be populated automatically by BaseComponent.__init_subclass__
REGISTERED_COMPONENT_TYPES: List[Type["BaseComponent"]] = []

# Initialize logger for this module
logger = logging.getLogger(__name__)


class BaseComponent(Base):
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

    # To guide dataclass for __init__ (init=False) and repr (repr=False)
    # The `entity` relationship is defined below using @declared_attr to correctly link via entity_id
    # entity: Mapped["Entity"] = field(init=False, repr=False) # This would be for dataclass field, SQLAlchemy handles the relationship

    @declared_attr
    def entity(cls) -> Mapped["Entity"]:
        """Relationship to the parent (owning) Entity."""
        # cls.entity_id refers to the entity_id column defined in this BaseComponent
        # (which will be part of each concrete component's table)
        return relationship(
            "Entity",
            foreign_keys=[cls.entity_id],  # type: ignore # Explicitly use the component's own entity_id for this link
            # back_populates="components", # REMOVED as Entity.components is removed
            repr=False,  # For SQLAlchemy's default repr of this relationship property
            init=False,  # Prevent 'entity' from being an __init__ parameter
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, entity_id={self.entity_id})"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Registers concrete component subclasses with the ecs_service.
        """
        super().__init_subclass__(**kwargs)
        # Avoid registering BaseComponent itself or other abstract classes if any
        is_abstract = cls.__dict__.get("__abstract__", False)
        logger.debug(f"BaseComponent.__init_subclass__ called for: {cls.__name__}, abstract: {is_abstract}")
        if not is_abstract:
            # Now appends to the list in this module, no service import needed here.
            if cls not in REGISTERED_COMPONENT_TYPES:
                REGISTERED_COMPONENT_TYPES.append(cls)
                logger.debug(f"Registered component: {cls.__name__}")
        else:
            logger.debug(f"Not registering abstract class: {cls.__name__}")
