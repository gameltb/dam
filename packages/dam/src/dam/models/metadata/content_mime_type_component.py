from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dam.models.core.base_component import UniqueComponent

if TYPE_CHECKING:
    from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent


@dataclass
class ContentMimeTypeComponent(UniqueComponent):
    __tablename__ = "component_content_mime_type"
    """
    A component that stores a reference to the mime type of an asset's content.
    """

    mime_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey("component_mime_type_concept.id"),
        nullable=False,
        index=True,
        comment="The ID of the MimeTypeConceptComponent that represents this mime type.",
    )

    mime_type_concept: Mapped["MimeTypeConceptComponent"] = relationship(
        "MimeTypeConceptComponent",
        lazy="joined",
        init=False,
    )
