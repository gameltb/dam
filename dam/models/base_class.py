from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

class Base(MappedAsDataclass, DeclarativeBase):
    """
    Base class for SQLAlchemy declarative models that are also dataclasses.
    """
    pass
