from dataclasses import dataclass, field
from datetime import datetime # Added import
from sqlalchemy import DateTime, text, func # Added func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base_class import Base # Updated import for Base
# from .types import PkId, TimestampCreated, TimestampUpdated # Commented out as not directly used by Entity now

# No need for @Base.mapped_as_dataclass here, as Base itself includes MappedAsDataclass
@dataclass
class Entity(Base): # Inherit from the new Base
    """
    Represents a unique digital asset in the system.
    An Entity is a container for various components that describe the asset.
    """
    __tablename__ = "entities"

    # Explicitly define all init=False fields using mapped_column directly
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, init=False)
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

    # Example of how relationships to components might be defined if needed directly on Entity
    # content_hashes: Mapped[list["ContentHashComponent"]] = relationship(
    #     back_populates="entity", cascade="all, delete-orphan"
    # )
    # perceptual_hashes: Mapped[list["ImagePerceptualHashComponent"]] = relationship(
    #     back_populates="entity", cascade="all, delete-orphan"
    # )

    # If using a BaseComponent with a backref, relationships might be implicitly available
    # or defined on the components themselves.

    def __repr__(self):
        return f"Entity(id={self.id})"
