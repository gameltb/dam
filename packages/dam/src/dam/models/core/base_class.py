from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(MappedAsDataclass, DeclarativeBase, kw_only=True):
    """
    Base class for SQLAlchemy declarative models that are also dataclasses.
    kw_only=True is applied globally by passing it in the inheritance list.
    """

    metadata = MetaData(naming_convention=naming_convention)
