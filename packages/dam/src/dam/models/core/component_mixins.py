"""SQLAlchemy mixins for component models."""

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declared_attr


class UniqueComponentMixin:
    """
    A mixin to mark a component as unique for an entity.

    This adds a database-level constraint.

    The `naming_convention` on the MetaData will give this constraint a
    table-specific name based on the `uq` key in the convention dictionary.
    """

    @declared_attr.directive
    def __table_args__(self) -> tuple[UniqueConstraint]:
        """Add a unique constraint on the entity_id column."""
        return (UniqueConstraint("entity_id"),)
