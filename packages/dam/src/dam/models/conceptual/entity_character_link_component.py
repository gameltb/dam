from typing import Optional

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.base_component import BaseComponent
from ..core.entity import Entity


class EntityCharacterLinkComponent(BaseComponent):
    """
    Links an entity (e.g., an image, a story) to a CharacterConceptComponent's entity,
    optionally defining the character's role or status in that context.
    """

    __tablename__ = "component_entity_character_link"

    # This is the Entity ID of the Entity that *has* the CharacterConceptComponent
    character_concept_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id"),  # Corrected FK to the 'entities' table
        nullable=False,
        index=True,
    )
    role_in_asset: Mapped[Optional[str]] = mapped_column(String(), nullable=True)

    # Relationship to the Entity that represents the character concept
    # This Entity is expected to have a CharacterConceptComponent attached to it.
    character_concept: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[character_concept_entity_id],  # Specify foreign_keys for clarity
        # If CharacterConceptComponent defines a backref, it can be added here.
        # e.g. back_populates="linked_assets"
    )

    # Ensure an entity cannot have the same character linked multiple times
    # unless the role makes it unique (e.g. character appears twice in different roles).
    # If role_in_asset can be NULL, this constraint might need adjustment or a DB-specific
    # way to handle NULLs in unique constraints if they are not treated as distinct.
    # For now, assuming (entity_id, character_concept_entity_id, role_in_asset) should be unique.
    __table_args__ = (
        UniqueConstraint(
            "entity_id",  # The entity being tagged with a character
            "character_concept_entity_id",  # The entity representing the character
            "role_in_asset",
            name="uq_entity_character_role",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"EntityCharacterLinkComponent(id={self.id}, entity_id={self.entity_id}, "
            f"character_concept_entity_id={self.character_concept_entity_id}, role='{self.role_in_asset}')"
        )
