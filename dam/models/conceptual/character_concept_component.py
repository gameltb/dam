from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text
from typing import Optional # Added Optional

from .base_conceptual_info_component import BaseConceptualInfoComponent
# from ..core.types import مفهوم_موجودیت_آی_دی # Assuming this is EntityId or similar - Removed for now, will use int


class CharacterConceptComponent(BaseConceptualInfoComponent):
    """
    Defines a character as a conceptual entity.
    """

    __tablename__ = "component_character_concept"

    # Inherits entity_id, concept_name, concept_description from BaseConceptualInfoComponent

    # Additional character-specific fields can be added here if needed,
    # for now, we'll rely on concept_name for character name and
    # concept_description for their bio/details.
    # Example:
    # species: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # abilities: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self):
        return (
            f"CharacterConceptComponent(id={self.id}, entity_id={self.entity_id}, "
            f"name='{self.concept_name}', description='{self.concept_description[:50] if self.concept_description else ''}...')"
        )
