import logging
from typing import Annotated, List

from sqlalchemy.orm import Mapped, mapped_column

from dam.core.stages import SystemStage
from dam.core.system_params import WorldContext, WorldSession
from dam.core.systems import system

# Marker component to indicate an entity needs audio processing.
# This can be added during ingestion or by other systems.
from dam.models.core import Entity
from dam.models.core.base_component import BaseComponent
from dam.models.properties import FilePropertiesComponent
from dam.models.semantic.audio_embedding_component import (
    get_audio_embedding_component_class,
)
from dam.services import audio_service, ecs_service

# Assuming this utility will be created or already exists and can provide a file path
from dam.utils.media_utils import get_file_path_for_entity


# Example Marker Component (Create this in a relevant models file if it doesn't exist)
class NeedsAudioProcessingMarker(BaseComponent):
    __tablename__ = "component_marker_needs_audio_processing"
    marker_set: Mapped[bool] = mapped_column(default=True)


logger = logging.getLogger(__name__)


@system(stage=SystemStage.CONTENT_ANALYSIS, auto_remove_marker=True)
async def audio_embedding_generation_system(
    session: WorldSession,
    world_context: WorldContext,  # For accessing world-specific config if needed
    # Get entities that have the NeedsAudioProcessingMarker
    entities_to_process: Annotated[List[Entity], "MarkedEntityList", NeedsAudioProcessingMarker],
):
    """
    Generates audio embeddings for entities marked with NeedsAudioProcessingMarker.
    This system iterates through entities that have been explicitly marked for audio processing.
    """
    if not entities_to_process:
        logger.info("AudioEmbeddingGenerationSystem: No entities marked for audio processing.")
        return

    logger.info(f"AudioEmbeddingGenerationSystem: Found {len(entities_to_process)} entities for audio processing.")

    # Could make the model configurable, e.g., via world_config or a resource
    default_model_name = audio_service.DEFAULT_AUDIO_MODEL_NAME
    processed_count = 0
    failed_count = 0

    for entity in entities_to_process:
        logger.debug(f"Processing entity {entity.id} for audio embedding.")

        # 1. Verify it's an audio file (optional, marker might be enough)
        file_props = await ecs_service.get_component(session, entity.id, FilePropertiesComponent)
        if not file_props or not file_props.mime_type or not file_props.mime_type.startswith("audio/"):
            logger.warning(
                f"Entity {entity.id} marked for audio processing, but not found or not an audio MIME type ({file_props.mime_type if file_props else 'N/A'}). Skipping."
            )
            continue

        # 2. Get the audio file path
        try:
            # This utility needs to be robust.
            # It might need access to FileStorageResource or similar from world_context's resource_manager
            # For now, it's a direct import.
            audio_path = await get_file_path_for_entity(
                session, entity.id, world_context.world_config.ASSET_STORAGE_PATH
            )
            if not audio_path:
                logger.error(f"Could not retrieve audio file path for entity {entity.id}. Skipping.")
                failed_count += 1
                continue
        except Exception as e:
            logger.error(f"Error getting audio file path for entity {entity.id}: {e}", exc_info=True)
            failed_count += 1
            continue

        # 3. Check if embedding already exists for the default model
        AudioEmbeddingClass = get_audio_embedding_component_class(default_model_name)
        if not AudioEmbeddingClass:
            logger.error(
                f"Could not find audio embedding class for model {default_model_name}. Skipping entity {entity.id}"
            )
            failed_count += 1
            continue

        existing_embeddings = await ecs_service.get_components_by_value(
            session, entity.id, AudioEmbeddingClass, attributes_values={"model_name": default_model_name}
        )
        if existing_embeddings:
            logger.info(
                f"Audio embedding for model {default_model_name} already exists for entity {entity.id}. Skipping generation."
            )
            # Marker will be removed automatically if auto_remove_marker=True
            processed_count += 1
            continue

        # 4. Generate and store embedding
        try:
            logger.info(
                f"Generating audio embedding for entity {entity.id} (path: {audio_path}) with model {default_model_name}."
            )
            embedding_component = await audio_service.generate_audio_embedding_for_entity(
                session,
                entity_id=entity.id,
                model_name=default_model_name,
                # model_params can be added if specific params are needed beyond default
                audio_file_path=audio_path,
            )
            if embedding_component:
                logger.info(
                    f"Successfully generated audio embedding for entity {entity.id} with model {default_model_name}."
                )
                processed_count += 1
            else:
                logger.warning(
                    f"Audio embedding generation returned None for entity {entity.id} with model {default_model_name}."
                )
                failed_count += 1
        except Exception as e:
            logger.error(f"Error generating audio embedding for entity {entity.id}: {e}", exc_info=True)
            failed_count += 1

    logger.info(
        f"AudioEmbeddingGenerationSystem: Finished processing. Processed: {processed_count}, Failed: {failed_count}."
    )


__all__ = ["audio_embedding_generation_system", "NeedsAudioProcessingMarker"]
