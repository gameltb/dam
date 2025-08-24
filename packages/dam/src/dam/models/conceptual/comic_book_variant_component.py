from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_variant_info_component import BaseVariantInfoComponent


class ComicBookVariantComponent(BaseVariantInfoComponent):
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

    is_primary_variant: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Indicates if this is the primary or preferred variant for the comic book concept.",
    )

    variant_description: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="A general description for this variant, e.g., 'Collector's Edition Scan', 'Digital Release'.",
    )

    # Could add other comic variant specific fields:
    # page_count: Mapped[int | None]
    # cover_image_entity_id: Mapped[int | None] (ForeignKey to another Entity that is the cover image)
    # edition_notes: Mapped[str | None]

    __table_args__ = (
        # Ensures that for a given comic book concept, you don't have two variants
        # described identically by language, format, and description (if description is key).
        # Adjust this constraint based on what truly makes a comic variant unique for a concept.
        # For example, if description isn't a good uniqueifier, remove it.
        # If multiple "English PDF" variants are allowed (e.g. different scans), then this constraint needs refinement
        # or an additional field like 'version_tag' or rely on distinct File Entities.
        UniqueConstraint(
            "conceptual_entity_id",
            "language",
            "format",
            "variant_description",  # This might be too specific for a unique constraint
            name="uq_comic_variant_details_per_concept",
        ),
        # An Entity (file) can only be one specific comic book variant of one comic book concept.
        # This is implicitly handled by BaseVariantInfoComponent having conceptual_entity_id
        # and the component being on a specific entity_id.
        # If further constraint is needed here, it would be similar to what was on the old VariantComponent:
        # UniqueConstraint("entity_id", "conceptual_entity_id", name="uq_comic_variant_entity_conceptual_entity"),
        # However, since entity_id is the PK of the component table (via BaseComponent), an entity can only have one
        # ComicBookVariantComponent row anyway. The main uniqueness is covered by the component type itself on an entity.
    )

    def __repr__(self):
        return (
            f"ComicBookVariantComponent(id={self.id}, entity_id={self.entity_id}, "
            f"conceptual_entity_id={self.conceptual_entity_id}, "
            f"lang='{self.language}', format='{self.format}', primary={self.is_primary_variant})"
        )
