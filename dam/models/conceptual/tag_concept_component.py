from sqlalchemy import String, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_conceptual_info_component import BaseConceptualInfoComponent
# It might be useful to have an Enum for scope types
# from enum import Enum
# class TagScopeType(Enum):
#     GLOBAL = "GLOBAL"
#     COMPONENT_CLASS_REQUIRED = "COMPONENT_CLASS_REQUIRED"
#     CONCEPTUAL_ASSET_LOCAL = "CONCEPTUAL_ASSET_LOCAL" # e.g. specific to a ComicBookConcept and its variants


class TagConceptComponent(BaseConceptualInfoComponent):
    """
    Defines a tag concept, its scope, and properties.
    This component is attached to an Entity that represents the tag itself.
    Other entities are then linked to this "Tag Entity" to apply the tag.
    """
    __tablename__ = "component_tag_concept"

    # id, entity_id, created_at, updated_at are inherited via BaseConceptualInfoComponent

    tag_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="The unique name of the tag (e.g., 'Sci-Fi', 'Needs Review', 'Artist:JohnDoe')."
    )

    # For simplicity, using strings for scope_type. An Enum could be used for stricter validation.
    tag_scope_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="GLOBAL", # Default to global scope
        index=True,
        comment="Defines the scope of the tag (e.g., 'GLOBAL', 'COMPONENT_CLASS_REQUIRED', 'CONCEPTUAL_ASSET_LOCAL')."
    )

    tag_scope_detail: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Details for the scope, e.g., component class name if scope_type is 'COMPONENT_CLASS_REQUIRED', "
                "or an Entity ID if scope_type is 'CONCEPTUAL_ASSET_LOCAL'."
    )

    tag_description: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="A description of what this tag represents or how it should be used."
    )

    allow_values: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="If True, this tag can be applied with an associated value (e.g., Tag 'Rating', Value '5 Stars'). "
                "If False, it's a simple label tag."
    )

    __table_args__ = (
        UniqueConstraint('tag_name', name='uq_tag_concept_name'),
        # If tag names should be unique only within a certain scope (e.g. different users can have same tag name),
        # this constraint would need to be more complex or handled at service layer.
        # For now, assuming tag_name is globally unique for simplicity of definition.
    )

    def __repr__(self):
        return (
            f"<TagConceptComponent id={self.id} entity_id={self.entity_id} "
            f"name='{self.tag_name}' scope='{self.tag_scope_type}'>"
        )
