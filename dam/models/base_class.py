from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass


class Base(MappedAsDataclass, DeclarativeBase, kw_only=True):
    """
    Base class for SQLAlchemy declarative models that are also dataclasses.
    kw_only=True is applied globally by passing it in the inheritance list.
    """

    pass
