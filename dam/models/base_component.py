from dataclasses import dataclass, field
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship

from .base_class import Base # Updated import for Base
from datetime import datetime # Added import
from sqlalchemy import DateTime, ForeignKey # text removed, func added
from sqlalchemy.sql import func # Added for func.now()
# PkId is still used by @declared_attr id, Timestamp types are not used here anymore
from .types import PkId
# Import Entity for ForeignKey relationship typing, will resolve with __future__.annotations if needed
# or by making it a string literal if type checking issues arise before full model setup.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .entity import Entity


# No need for @Base.mapped_as_dataclass here
@dataclass
class BaseComponent(Base): # Inherit from the new Base
    """
    Abstract base class for all components.
    Provides common fields like id, entity_id (linking to an Entity),
    and timestamps.
    """
    __abstract__ = True # This class will not be mapped to its own table

    # Using declared_attr for fields that might need table-specific context,
    # though for simple fields like id, created_at, updated_at, direct Mapped annotation is fine.
    # For entity_id, declared_attr ensures it's correctly set up for each subclass's table.

    @declared_attr
    def id(cls) -> Mapped[PkId]: # PkId type carries primary_key=True etc.
        return mapped_column(init=False) # init=False is the dataclass control part

    @declared_attr
    def entity_id(cls) -> Mapped[int]:
        # Note: ForeignKey uses the actual table name and column name.
        # This field should be an __init__ arg for subclasses, so no init=False.
        return mapped_column(ForeignKey("entities.id"), index=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(), # Changed to func.now()
        nullable=False,
        init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(), # Changed to func.now()
        onupdate=func.now(),      # Changed to func.now()
        nullable=False,
        init=False
    )

    @declared_attr
    def entity(cls) -> Mapped["Entity"]:
        """Relationship to the parent Entity."""
        return relationship("Entity", repr=False)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, entity_id={self.entity_id})"
