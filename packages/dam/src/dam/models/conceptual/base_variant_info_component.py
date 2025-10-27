"""Base classes for variant asset components."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship  # Added declared_attr

from ..core.base_component import BaseComponent, UniqueComponent

if TYPE_CHECKING:
    from dam.models.core.entity import Entity


class BaseVariantInfoComponent(BaseComponent):
    """
    Abstract base class for components that link a concrete File Entity.

    (a specific file manifestation) to a Conceptual Asset Entity.
    This component defines that the File Entity is a 'variant' of a conceptual work.

    Concrete subclasses will define attributes specific to being a variant
    of a particular type of conceptual asset (e.g., ComicBookVariantComponent
    would define language, format, etc.).
    """

    __abstract__ = True

    # id, entity_id are inherited from BaseComponent.
    # entity_id here refers to the Entity that *is* this variant (the File Entity).

    conceptual_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
        init=False,  # This field is populated via the 'conceptual_asset' relationship
        comment="The ID of the Entity that represents the Conceptual Asset this variant belongs to.",
    )

    @declared_attr
    def conceptual_asset(self) -> Mapped["Entity"]:
        """Relationship to the parent (owning) Conceptual Asset Entity."""
        return relationship(
            "Entity",
            foreign_keys=[self.conceptual_entity_id],  # type: ignore # Use cls.conceptual_entity_id
            # backref="variants_conceptual_links", # Consider if a backref is needed and how it would work with multiple variant types
            repr=False,
            init=False,  # Prevent 'conceptual_asset' from being an __init__ parameter
        )


class UniqueBaseVariantInfoComponent(UniqueComponent):
    """
    Abstract base class for unique components that link a concrete File Entity.

    to a Conceptual Asset Entity.
    """

    __abstract__ = True

    conceptual_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
        init=False,
        comment="The ID of the Entity that represents the Conceptual Asset this variant belongs to.",
    )

    @declared_attr
    def conceptual_asset(self) -> Mapped["Entity"]:
        """Relationship to the parent (owning) Conceptual Asset Entity."""
        return relationship(
            "Entity",
            foreign_keys=[self.conceptual_entity_id],  # type: ignore
            repr=False,
            init=False,
        )
