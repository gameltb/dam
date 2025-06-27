from datetime import datetime  # Added import

from sqlalchemy import DateTime, func  # Added func
from sqlalchemy.orm import Mapped, mapped_column, relationship # Added relationship

from .base_class import Base  # Updated import for Base
from .base_component import BaseComponent # Ensure BaseComponent is imported for type hints

# from .types import PkId, TimestampCreated, TimestampUpdated
# Commented out as not directly used by Entity now


# No need for @Base.mapped_as_dataclass here, as Base itself includes MappedAsDataclass
# kw_only=True and @dataclass behavior are inherited from Base
class Entity(Base):  # Inherit from the new Base
    """
    Represents a unique digital asset in the system.
    An Entity is a container for various components that describe the asset.
    """

    __tablename__ = "entities"

    # Explicitly define all init=False fields using mapped_column directly
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

    # Example of how relationships to components might be defined if needed
    # directly on Entity
    # content_hashes: Mapped[list["ContentHashComponent"]] = relationship(
    #     back_populates="entity", cascade="all, delete-orphan"
    # )
    # perceptual_hashes: Mapped[list["ImagePerceptualHashComponent"]] = relationship(
    #     back_populates="entity", cascade="all, delete-orphan"
    # )

    # If using a BaseComponent with a backref, relationships might be
    # implicitly available or defined on the components themselves.

    # This 'components' relationship is primarily to satisfy `back_populates`
    # from BaseComponent.entity. Querying this directly might be complex due to
    # BaseComponent being abstract. Specific component relationships are usually preferred.
    from typing import List
    from dataclasses import field as dataclass_field
    # BaseComponent is already imported at the top of the file

    # Define 'components' for dataclass __init__ behavior (not an init arg, defaults to empty list)
    components: Mapped[List["BaseComponent"]] = dataclass_field(
        default_factory=list, init=False
    )

    # Explicitly map the 'components' attribute to the SQLAlchemy relationship
    # This pattern is used when the default MappedAsDataclass behavior for relationships
    # (expecting init=False automatically) conflicts with other dataclass settings like kw_only=True.
    __mapper_args__ = {
        "properties": {
            "components": relationship(
                BaseComponent, # Use direct class reference
                back_populates="entity",
                cascade="all, delete-orphan",
            )
        }
    }

    def __repr__(self):
        return f"Entity(id={self.id})"
