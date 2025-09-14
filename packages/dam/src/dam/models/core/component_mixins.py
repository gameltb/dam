from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declared_attr


class UniqueComponentMixin:
    """
    A mixin to mark a component as unique for an entity by adding a
    database-level constraint using the @declared_attr pattern.
    """

    @declared_attr.directive
    def __table_args__(cls):
        """
        This method is called by SQLAlchemy during the class mapping process.
        It returns a tuple containing the UniqueConstraint.
        """
        return (UniqueConstraint("entity_id", name=f"uq_{cls.__tablename__}_entity_id"),)
