from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..core.component_mixins import UniqueComponentMixin
from .base_variant_info_component import BaseVariantInfoComponent


class ComicBookVariantComponent(UniqueComponentMixin, BaseVariantInfoComponent):
    """
    Represents a specific variant of a Comic Book Concept.
    This component is attached to a File Entity and links it to a
    ComicBookConceptComponent's Entity.
    """

    __tablename__ = "component_comic_book_variant"

    # id, entity_id, conceptual_entity_id are inherited.

    language: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True, comment="Language of this comic book variant (e.g., 'en', 'jp')."
    )

    format: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True, comment="File format of this variant (e.g., 'PDF', 'CBZ', 'ePub')."
    )

    scan_quality: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Description of scan quality, if applicable (e.g., '300dpi', 'WebRip', 'Archive Grade').",
    )

    variant_description: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="A general description for this variant, e.g., 'Collector's Edition Scan', 'Digital Release'.",
    )

    is_primary_variant: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Indicates if this is the primary or preferred variant for the comic book concept.",
    )

    __table_args__ = UniqueComponentMixin.__table_args__ + (  # type: ignore
        UniqueConstraint(
            "conceptual_entity_id",
            "language",
            "format",
            "variant_description",  # This might be too specific for a unique constraint
            name="uq_comic_variant_details_per_concept",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"ComicBookVariantComponent(id={self.id}, entity_id={self.entity_id}, "
            f"conceptual_entity_id={self.conceptual_entity_id}, "
            f"lang='{self.language}', format='{self.format}', primary={self.is_primary_variant})"
        )
