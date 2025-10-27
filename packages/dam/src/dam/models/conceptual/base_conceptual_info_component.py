"""Base classes for conceptual asset components."""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import BaseComponent, UniqueComponent


class BaseConceptualInfoComponent(BaseComponent):
    """
    Abstract base class for components that define a 'Conceptual Asset'.

    A Conceptual Asset represents an abstract idea or work, which can have
    multiple concrete versions or manifestations (Variants).

    Concrete subclasses will define the specific attributes that characterize
    a particular type of conceptual asset (e.g., ComicBookConceptComponent
    would define series_title, issue_number, etc.).
    """

    __abstract__ = True

    concept_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    concept_description: Mapped[str | None] = mapped_column(Text, nullable=True)



class UniqueBaseConceptualInfoComponent(UniqueComponent):
    """Abstract base class for unique components that define a 'Conceptual Asset'."""

    __abstract__ = True

    concept_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    concept_description: Mapped[str | None] = mapped_column(Text, nullable=True)
