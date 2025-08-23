"""
Defines "marker components" used in the Entity-Component-System (ECS) architecture.

Marker components are special types of components that typically do not hold any data themselves,
beyond what `BaseComponent` provides (like `id`, `entity_id`, and timestamps).
Their primary purpose is to "mark" or "tag" an entity, signaling that it needs
a particular type of processing or that a certain state has been reached.

Systems can then query for entities that have a specific marker component attached
(e.g., using `MarkedEntityList[SomeMarkerComponent]` as a system parameter).
After processing, the system (or the scheduler) might remove the marker component
or add a different one (e.g., `ProcessingCompleteMarker`).

This approach allows for decoupled workflows where different systems can react to
the state of entities as indicated by these markers.
"""

from sqlalchemy.orm import Mapped, mapped_column
from dam.models.core.base_component import BaseComponent

# No specific fields needed for marker components, they exist by their type.


class NeedsMetadataExtractionComponent(BaseComponent):
    """
    A marker component indicating that an entity requires metadata extraction.
    This component will be added by asset_service after initial ingestion
    and removed by the MetadataExtractionSystem after processing.
    """

    __tablename__ = "component_marker_needs_metadata_extraction"

    # No additional fields needed beyond what BaseComponent provides (id, entity_id, timestamps)
    # The presence of this component on an entity is its data.

    def __repr__(self):
        return f"NeedsMetadataExtractionComponent(id={self.id}, entity_id={self.entity_id})"


class MetadataExtractedComponent(BaseComponent):
    """
    A marker component indicating that an entity has had its metadata extracted.
    Can be used to prevent re-processing or to query for processed entities.
    """

    __tablename__ = "component_marker_metadata_extracted"

    def __repr__(self):
        return f"MetadataExtractedComponent(id={self.id}, entity_id={self.entity_id})"


# Add other marker components here as needed for different systems/pipelines.
# For example:
# class NeedsThumbnailGenerationComponent(BaseComponent): pass
# class NeedsTranscodingComponent(BaseComponent): pass
# class NeedsAIDescriptionComponent(BaseComponent): pass

class NeedsAudioProcessingMarker(BaseComponent):
    __tablename__ = "component_marker_needs_audio_processing"
    marker_set: Mapped[bool] = mapped_column(default=True)

class NeedsAutoTaggingMarker(BaseComponent):
    __tablename__ = "component_marker_needs_auto_tagging"
    pass

class AutoTaggingCompleteMarker(BaseComponent):
    __tablename__ = "component_marker_auto_tagging_complete"
    pass
