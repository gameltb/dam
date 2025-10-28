"""Data model for character concepts."""

from .base_conceptual_info_component import UniqueBaseConceptualInfoComponent


class CharacterConceptComponent(UniqueBaseConceptualInfoComponent):
    """Defines a character as a conceptual entity."""

    __tablename__ = "component_character_concept"

    # Inherits entity_id, concept_name, concept_description from BaseConceptualInfoComponent

    # Additional character-specific fields can be added here if needed,
    # for now, we'll rely on concept_name for character name and
    # concept_description for their bio/details.
    # Example:
    # species: Mapped[Optional[str]] = mapped_column(String(), nullable=True)
    # abilities: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
