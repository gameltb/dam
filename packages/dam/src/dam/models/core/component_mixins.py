from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declared_attr


class UniqueComponentMixin:
    """
    A mixin to mark a component as unique for an entity by adding a
    database-level constraint.

    The `naming_convention` on the MetaData will give this constraint a
    table-specific name based on the `uq` key in the convention dictionary.
    """

    @declared_attr.directive
    def __table_args__(cls):
        return (UniqueConstraint("entity_id"),)
