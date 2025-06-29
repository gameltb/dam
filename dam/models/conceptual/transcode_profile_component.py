from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

from dam.models.core.base_component import BaseComponent
from dam.models.conceptual.base_conceptual_info_component import BaseConceptualInfoComponent
from sqlalchemy import ForeignKey # Added for ForeignKey


class TranscodeProfileComponent(BaseConceptualInfoComponent): # Removed BaseComponent
    """
    Component defining a transcoding profile.
    This is a conceptual asset, meaning an entity with this component
    represents the concept of a specific transcoding configuration.
    """
    __tablename__ = "transcode_profiles"

    # This 'id' is the primary key of this table AND a foreign key to entities.id
    id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)

    profile_name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "ffmpeg", "cjxl", "avifenc"
    parameters: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "-crf 23 -preset medium"
    output_format: Mapped[str] = mapped_column(String, nullable=False) # e.g., "avif", "jxl", "mp4"
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    # For BaseConceptualInfoComponent
    concept_name: Mapped[str] = mapped_column(String, nullable=False) # Will be same as profile_name
    concept_description: Mapped[str | None] = mapped_column(String, nullable=True) # Will be same as description

    __mapper_args__ = {
        "polymorphic_identity": "transcode_profile",
        # For joined table, inherit_condition is often not needed if PK/FK is clear.
        # SQLAlchemy should be able to link TranscodeProfileComponent.id to Entity.id
        # via the BaseComponent's mapping of its 'entity_id' (which becomes this table's 'id').
    }

    def __repr__(self) -> str:
        return (
            f"<TranscodeProfileComponent(id={self.id}, entity_id={self.entity_id}, "
            f"profile_name='{self.profile_name}', tool='{self.tool_name}', format='{self.output_format}')>"
        )
