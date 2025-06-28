from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

from dam.models.core.base_component import BaseComponent
from dam.models.conceptual.base_conceptual_info_component import BaseConceptualInfoComponent


class TranscodeProfileComponent(BaseConceptualInfoComponent, BaseComponent):
    """
    Component defining a transcoding profile.
    This is a conceptual asset, meaning an entity with this component
    represents the concept of a specific transcoding configuration.
    """
    __tablename__ = "transcode_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    entity_id: Mapped[int] = mapped_column(nullable=False, index=True, unique=True) # Foreign key to entities.id will be in __mapper_args__

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
        "inherit_condition": id == BaseComponent.id, # type: ignore
         # Link to Entity table via BaseComponent's entity_id if not already handled by BaseComponent's own setup
        "primary_key": [id], # Explicitly define primary key if not automatically picked up
    }

    def __repr__(self) -> str:
        return (
            f"<TranscodeProfileComponent(id={self.id}, entity_id={self.entity_id}, "
            f"profile_name='{self.profile_name}', tool='{self.tool_name}', format='{self.output_format}')>"
        )
