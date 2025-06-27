import logging  # Added import
from datetime import datetime  # Added import

# PkId is still used by @declared_attr id, Timestamp types are not used here anymore
# Import Entity for ForeignKey relationship typing, will resolve with
# __future__.annotations if needed or by making it a string literal if type
# checking issues arise before full model setup.
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,  # text removed, func added
    ForeignKey,
)
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship
from sqlalchemy.sql import func  # Added for func.now()

from .base_class import Base  # Updated import for Base

if TYPE_CHECKING:
    from .entity import Entity


from typing import List, Type  # For List and Type hints

# List to hold all registered component types that inherit from BaseComponent.
# This list will be populated automatically by BaseComponent.__init_subclass__
REGISTERED_COMPONENT_TYPES: List[Type["BaseComponent"]] = []

# Initialize logger for this module
logger = logging.getLogger(__name__)


# No need for @Base.mapped_as_dataclass here
# kw_only=True and @dataclass behavior are inherited from Base
class BaseComponent(Base):  # Inherit from the new Base
    """
    Abstract base class for all components.
    Provides common fields like id, entity_id (linking to an Entity),
    and timestamps.
    """

    __abstract__ = True  # This class will not be mapped to its own table

    # Using declared_attr for fields that might need table-specific context,
    # though for simple fields like id, created_at, updated_at, direct Mapped
    # annotation is fine. For entity_id, declared_attr ensures it's correctly
    # set up for each subclass's table.

    # Attributes that are __init__ parameters should come first.
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True, nullable=False) # Removed init=False

    # Attributes that are NOT __init__ parameters.
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, init=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # Changed to func.now()
        nullable=False,
        init=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # Changed to func.now()
        onupdate=func.now(),  # Changed to func.now()
        nullable=False,
        init=False,
    )

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
            foreign_keys=[cls.entity_id],  # Explicitly use the component's own entity_id for this link
            # back_populates="components",  # Removed as Entity.components is removed
            repr=False,  # For SQLAlchemy's default repr of this relationship property
            init=False, # Explicitly set init=False for the relationship
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, entity_id={self.entity_id})"

    def __init_subclass__(cls, **kwargs):
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
