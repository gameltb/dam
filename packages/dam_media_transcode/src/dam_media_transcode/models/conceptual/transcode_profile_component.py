"""Defines the TranscodeProfileComponent for representing a transcoding profile."""
from typing import ClassVar

from dam.models.conceptual.base_conceptual_info_component import BaseConceptualInfoComponent
from sqlalchemy import (
    ForeignKey,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column


class TranscodeProfileComponent(BaseConceptualInfoComponent):
    """
    Component defining a transcoding profile.

    This is a conceptual asset, meaning an entity with this component
    represents the concept of a specific transcoding configuration.
    """

    __tablename__ = "component_transcode_profile"

    id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)

    profile_name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    parameters: Mapped[str] = mapped_column(String, nullable=False)
    output_format: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    concept_name: Mapped[str] = mapped_column(String, nullable=False)
    concept_description: Mapped[str | None] = mapped_column(String, nullable=True)

    __mapper_args__: ClassVar[dict[str, str]] = {
        "polymorphic_identity": "transcode_profile",
    }

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return (
            f"<TranscodeProfileComponent(id={self.id}, entity_id={self.entity_id}, "
            f"profile_name='{self.profile_name}', tool='{self.tool_name}', format='{self.output_format}')>"
        )
