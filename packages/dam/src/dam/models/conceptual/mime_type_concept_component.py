from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_conceptual_info_component import BaseConceptualInfoComponent


class MimeTypeConceptComponent(BaseConceptualInfoComponent):
    """
    Defines a mime type concept.
    This component is attached to an Entity that represents the mime type itself.
    Other entities can then link to this "Mime Type Entity" to refer to the mime type.
    """

    __tablename__ = "component_mime_type_concept"

    mime_type: Mapped[str] = mapped_column(
        String(),
        nullable=False,
        index=True,
        comment="The unique mime type string (e.g., 'image/png', 'application/pdf').",
    )

    __table_args__ = (UniqueConstraint("mime_type", name="uq_mime_type_concept_mime_type"),)

    def __repr__(self) -> str:
        return f"<MimeTypeConceptComponent id={self.id} entity_id={self.entity_id} mime_type='{self.mime_type}'>"
