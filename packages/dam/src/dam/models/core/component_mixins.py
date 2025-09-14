from sqlalchemy import UniqueConstraint


class UniqueComponentMixin:
    """
    A mixin to mark a component as unique for an entity by adding a
    database-level constraint.

    The `naming_convention` on the MetaData will give this constraint a
    table-specific name based on the `uq` key in the convention dictionary.
    """

    __table_args__ = (UniqueConstraint("entity_id"),)
