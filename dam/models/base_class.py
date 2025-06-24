from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

class Base(MappedAsDataclass, DeclarativeBase):
    """
    Base class for SQLAlchemy declarative models that are also dataclasses.
    """
    # __dataclass_args__ removed, kw_only will be applied per model
    pass
