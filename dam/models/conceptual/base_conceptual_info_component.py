from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text
from typing import Optional

from ..core.base_component import BaseComponent


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
    concept_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self):
        # Since this is abstract, direct instantiation isn't typical for __repr__
        # but subclasses might call super().__repr__()
        return f"{self.__class__.__name__}(id={self.id}, entity_id={self.entity_id}, concept_name='{self.concept_name}')"
