import uuid
from pathlib import Path
from typing import Optional, Tuple  # Added List, Dict, Any for type hints potentially used by events

from sqlalchemy.future import select
from sqlalchemy.orm import Session, joinedload

from dam.core.config import settings  # Import the global settings instance

# Updated event imports: BaseEvent is needed, others are now defined in core.events
from dam.core.events import (
    AssetFileIngestionRequested,
)
from dam.core.world import World
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.conceptual.transcoded_variant_component import TranscodedVariantComponent
from dam.models.core.entity import Entity
from dam.models.core.file_location_component import FileLocationComponent
from dam.models.properties.file_properties_component import FilePropertiesComponent  # Added import
from dam.services import (
    ecs_service,
    file_operations,
    tag_service,
)

# Removed world_service as it's not directly used here.
from dam.utils.media_utils import TranscodeError, transcode_media


class TranscodeServiceError(Exception):
    """Custom exception for TranscodeService errors."""

    pass


async def create_transcode_profile(
    world: World,
    profile_name: str,
    tool_name: str,
    parameters: str,
    output_format: str,
    description: Optional[str] = None,
) -> Entity:
    """
    Creates a new transcoding profile as a conceptual asset.
    """
    async with world.db_session_maker() as session:
        # Check if profile with this name already exists
        stmt_existing = select(TranscodeProfileComponent).where(TranscodeProfileComponent.profile_name == profile_name)
        existing_profile = (await session.execute(stmt_existing)).scalars().first()
        if existing_profile:
            raise TranscodeServiceError(f"Transcode profile '{profile_name}' already exists.")

        # Create a new entity for this conceptual asset
        profile_entity = Entity()
        session.add(profile_entity)
        await session.flush()  # To get profile_entity.id

        # Create the TranscodeProfileComponent
        # For BaseConceptualInfoComponent fields:
        concept_name = profile_name
        concept_description = description

        profile_component = TranscodeProfileComponent(
            id=profile_entity.id,  # For TranscodeProfileComponent's own PK/FK 'id' field
            profile_name=profile_name,
            tool_name=tool_name,
            parameters=parameters,
            output_format=output_format,
            description=description,
            concept_name=concept_name,
            concept_description=concept_description,
        )
        await ecs_service.add_component_to_entity(session, profile_entity.id, profile_component)

        # Add a tag to mark this entity as a "Transcode Profile"
        # This uses the existing tag_service
        try:
            tag_concept_name = "System:TranscodeProfile"
            # Ensure the tag concept exists
            try:
                tag_concept_entity = await tag_service.get_tag_concept_by_name(session, tag_concept_name)
            except tag_service.TagConceptNotFoundError:
                tag_concept_entity = await tag_service.create_tag_concept(
                    session,
                    tag_name=tag_concept_name,
                    description="Marks an entity as a transcoding profile.",
                    scope_type="GLOBAL",  # Or a more specific scope if desired
                )

            await tag_service.apply_tag_to_entity(
                session,  # Pass session directly
                entity_id_to_tag=profile_entity.id,
                tag_concept_entity_id=tag_concept_entity.id,  # Corrected parameter name
            )
        except Exception as e:
            # Log this error, but don't let it fail profile creation entirely
            # Or re-raise if this tag is critical
            world.logger.warning(
                f"Could not apply system tag to transcode profile '{profile_name}': {e}", exc_info=True
            )

        await session.commit()
        await session.refresh(profile_entity)
        await session.refresh(profile_component)

        print(f"Transcode profile '{profile_name}' (Entity ID: {profile_entity.id}) created successfully.")
        return profile_entity


async def get_transcode_profile_by_name_or_id(
    world: World, profile_identifier: str | int, session: Optional[Session] = None
) -> Tuple[Entity, TranscodeProfileComponent]:
    """
    Retrieves a transcode profile entity and its component by name or entity ID.
    """

    async def _get(db_session: Session):
        if isinstance(profile_identifier, int):  # It's an entity ID
            stmt = (
                select(Entity, TranscodeProfileComponent)
                .join(TranscodeProfileComponent, Entity.id == TranscodeProfileComponent.entity_id)
                .where(Entity.id == profile_identifier)
            )
        else:  # It's a profile name
            stmt = (
                select(Entity, TranscodeProfileComponent)
                .join(TranscodeProfileComponent, Entity.id == TranscodeProfileComponent.entity_id)
                .where(TranscodeProfileComponent.profile_name == profile_identifier)
            )

        result = (await db_session.execute(stmt)).first()
        if not result:
            raise TranscodeServiceError(f"Transcode profile '{profile_identifier}' not found.")
        return result[0], result[1]  # Entity, TranscodeProfileComponent

    if session:
        return await _get(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _get(new_session)


async def _get_source_asset_filepath(world: World, asset_entity_id: int, session: Session) -> Path:
    """Helper to get a readable filepath for a source asset."""
    # Try to get FileLocationComponent for physical path
    flc = await ecs_service.get_component(
        session,
        entity_id=asset_entity_id,
        component_type=FileLocationComponent,  # type: ignore
    )
    if not flc or not flc.physical_path_or_key:
        raise TranscodeServiceError(f"Asset entity {asset_entity_id} has no readable FileLocationComponent.")

    source_path = Path(flc.physical_path_or_key)
    if not source_path.is_file() or not source_path.exists():
        # Check if it's a relative path to storage base
        from dam.resources.file_storage_resource import FileStorageResource  # Import the type

        storage_resource = world.get_resource(FileStorageResource)
        if storage_resource:
            # This assumes FileStorageResource has a base_path attribute (corrected: use world.config.ASSET_STORAGE_PATH)
            # And the physical_path_or_key might be relative to it
            # This logic might need to be more robust depending on how FileStorageResource works
            potential_path = Path(world.config.ASSET_STORAGE_PATH) / flc.contextual_filename  # type: ignore
            if potential_path.exists() and potential_path.is_file():
                source_path = potential_path
            else:  # Check content_identifier based path
                # Use get_file_path which takes the content_identifier (hash)
                content_id_path = storage_resource.get_file_path(flc.content_identifier)  # type: ignore
                if (
                    content_id_path and content_id_path.exists() and content_id_path.is_file()
                ):  # get_file_path can return None
                    source_path = content_id_path
                else:
                    raise TranscodeServiceError(
                        f"Source file for asset {asset_entity_id} at {flc.physical_path_or_key} or {potential_path} or {content_id_path} not found or not a file."
                    )
        else:  # This case means storage_resource itself was None, though get_resource should raise if not found.
            raise TranscodeServiceError(
                f"Source file for asset {asset_entity_id} at {flc.physical_path_or_key} not found and FileStorageResource could not be obtained to resolve further."
            )

    return source_path


async def apply_transcode_profile(
    world: World,
    source_asset_entity_id: int,
    profile_entity_id: int,  # Can also be profile name, handled by get_transcode_profile
    output_parent_dir: Optional[Path] = None,  # If None, use default temp/cache location from settings
) -> Entity:
    """
    Applies a transcoding profile to a source asset, creates a new asset for the
    transcoded file, and links them.
    """
    async with world.db_session_maker() as session:
        # 1. Get Transcode Profile
        _profile_entity, profile_component = await get_transcode_profile_by_name_or_id(
            world, profile_entity_id, session=session
        )

        # 2. Get Source Asset's File Path
        source_entity = await ecs_service.get_entity(session, source_asset_entity_id)
        if not source_entity:
            raise TranscodeServiceError(f"Source asset entity ID {source_asset_entity_id} not found.")

        source_filepath = await _get_source_asset_filepath(world, source_asset_entity_id, session)

        # Determine original filename for the new asset based on source, if possible
        source_fpc = await ecs_service.get_component(session, source_asset_entity_id, FilePropertiesComponent)  # type: ignore
        original_filename_base = "transcoded_file"
        if source_fpc and source_fpc.original_filename:  # type: ignore
            original_filename_base = Path(source_fpc.original_filename).stem  # type: ignore

        new_asset_original_filename = f"{original_filename_base}_{profile_component.profile_name.replace(' ', '_')}.{profile_component.output_format}"

        # 3. Determine Output Path for Transcoded File
        # Use a temporary/cache directory from settings or a specific output_parent_dir
        # The actual filename will be unique.
        # settings instance is imported directly now
        temp_transcode_dir = Path(settings.TRANSCODING_TEMP_DIR)
        temp_transcode_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

        if output_parent_dir:
            final_output_dir_base = output_parent_dir
        else:
            # If no specific output path, the file will be ingested into DAM's content-addressable storage.
            # The transcode_media utility needs a temporary place to write the file before ingestion.
            final_output_dir_base = temp_transcode_dir

        final_output_dir_base.mkdir(parents=True, exist_ok=True)

        # Generate a unique name for the temporary output file before ingestion
        unique_suffix = uuid.uuid4().hex[:8]
        temp_output_filename = f"{Path(source_filepath).stem}_{profile_component.profile_name.replace(' ', '_')}_{unique_suffix}.{profile_component.output_format}"
        temp_output_filepath = final_output_dir_base / temp_output_filename

        # 4. Execute Transcoding
        print(
            f"Applying profile '{profile_component.profile_name}' to asset ID {source_asset_entity_id} ({source_filepath})"
        )
        print(f"Output will be temporarily written to: {temp_output_filepath}")

        try:
            transcoded_filepath = await transcode_media(  # Added await
                input_path=source_filepath,
                output_path=temp_output_filepath,
                tool_name=profile_component.tool_name,
                tool_params=profile_component.parameters,
            )
        except TranscodeError as e:
            raise TranscodeServiceError(f"Transcoding failed: {e}") from e
        except FileNotFoundError as e:  # e.g. input file gone missing
            raise TranscodeServiceError(f"Transcoding input file error: {e}") from e

        if not transcoded_filepath.exists() or transcoded_filepath.stat().st_size == 0:
            # Should be caught by transcode_media, but as a safeguard:
            if transcoded_filepath.exists():
                transcoded_filepath.unlink(missing_ok=True)
            raise TranscodeServiceError(f"Transcoding produced no output or an empty file at {transcoded_filepath}.")

        # 5. Ingest the Transcoded File as a New Asset
        # This uses the existing asset ingestion event flow.
        # The ingestion system will calculate hashes, extract metadata, and create FileLocationComponent.
        try:
            _ret_original_filename, ret_size_bytes, ret_mime_type = file_operations.get_file_properties(
                transcoded_filepath
            )
        except Exception as e:
            if transcoded_filepath.exists():  # Check if it exists before unlinking
                transcoded_filepath.unlink(missing_ok=True)  # Clean up temp file
            raise TranscodeServiceError(f"Could not get properties of transcoded file {transcoded_filepath}: {e}")

        ingestion_event = AssetFileIngestionRequested(
            filepath_on_disk=transcoded_filepath,
            original_filename=new_asset_original_filename,  # Use the derived meaningful name
            mime_type=ret_mime_type,
            size_bytes=ret_size_bytes,
            world_name=world.name,
            # We could add custom metadata here if the event supports it, e.g., source_asset_id
            # custom_metadata={"source_for_transcode_of_entity_id": source_asset_entity_id}
        )

        # Dispatch event and wait for ingestion to complete (or at least for entity creation)
        # This requires the event handler for AssetFileIngestionRequested to be robust
        # and ideally to provide the new entity ID back, or we query for it by hash.

        # For now, let's assume the event handler will create the entity and its core components.
        # We'll then find the entity by its hash to link it.

        await world.dispatch_event(ingestion_event)

        # To ensure the file is processed by ingestion systems (metadata, etc.)
        # This would typically run after the event that adds NeedsMetadataExtractionComponent
        await world.execute_stage(SystemStage.METADATA_EXTRACTION)  # type: ignore

        # Find the newly ingested asset by its hash.
        # The ingestion system should have added ContentHashSHA256Component.
        transcoded_file_sha256_bytes = bytes.fromhex(
            file_operations.calculate_sha256(transcoded_filepath)
        )  # Ensure bytes

        # Query for the entity with this SHA256 hash
        # Use ecs_service.find_entity_by_content_hash
        newly_ingested_entity_result = await ecs_service.find_entity_by_content_hash(
            session, transcoded_file_sha256_bytes, "sha256"
        )

        if not newly_ingested_entity_result:
            transcoded_filepath.unlink(missing_ok=True)  # Clean up temp file
            # This could happen if ingestion failed silently or hash mismatch.
            raise TranscodeServiceError(
                f"Failed to find newly ingested transcoded asset with SHA256 {transcoded_file_sha256_bytes.hex()}. "
                "Ingestion might have failed or hash calculation mismatch."
            )

        transcoded_asset_entity = newly_ingested_entity_result
        print(f"Transcoded asset ingested. New Entity ID: {transcoded_asset_entity.id}")

        # 6. Create and Attach TranscodedVariantComponent
        transcoded_variant_comp = TranscodedVariantComponent(
            original_asset_entity_id=source_asset_entity_id,
            transcode_profile_entity_id=profile_component.entity_id,  # Use entity_id from profile_component
            transcoded_file_size_bytes=transcoded_filepath.stat().st_size,
            quality_metric_vmaf=None,
            quality_metric_ssim=None,
            custom_metrics_json=None,  # Added field
        )
        await ecs_service.add_component_to_entity(session, transcoded_asset_entity.id, transcoded_variant_comp)

        # 7. Commit changes (TranscodedVariantComponent)
        # The ingestion event would have handled its own commit for the new asset.
        # This commit is for the TranscodedVariantComponent.
        await session.commit()
        await session.refresh(transcoded_asset_entity)  # Refresh to see all components if needed

        print(f"TranscodedVariantComponent created and linked for new asset entity ID {transcoded_asset_entity.id}.")

        # 8. Clean up the temporary transcoded file if it's different from final DAM storage path
        # The AssetFileIngestionRequested event handler should specify if the file was copied or moved.
        # If it was copied, we should delete the temp_output_filepath.
        # Assuming the ingestion process copies the file to DAM storage:
        if transcoded_filepath.exists():
            try:
                transcoded_filepath.unlink()
                print(f"Cleaned up temporary transcoded file: {transcoded_filepath}")
            except OSError as e:
                print(f"Warning: Could not delete temporary transcoded file {transcoded_filepath}: {e}")

        return transcoded_asset_entity


async def get_transcoded_variants_for_original(
    world: World, original_asset_entity_id: int, session: Optional[Session] = None
) -> list[Tuple[Entity, TranscodedVariantComponent, TranscodeProfileComponent]]:
    """
    Retrieves all transcoded variants for a given original asset entity.
    Returns a list of tuples: (transcoded_entity, variant_component, profile_component).
    """

    async def _get(db_session: Session):
        stmt = (
            select(Entity, TranscodedVariantComponent, TranscodeProfileComponent)
            .join(TranscodedVariantComponent, Entity.id == TranscodedVariantComponent.entity_id)
            .join(
                TranscodeProfileComponent,
                TranscodedVariantComponent.transcode_profile_entity_id == TranscodeProfileComponent.entity_id,
            )
            .where(TranscodedVariantComponent.original_asset_entity_id == original_asset_entity_id)
            .options(
                joinedload(Entity.components_collection)  # Ensure components are loaded for the transcoded entity
            )
        )
        results = (await db_session.execute(stmt)).all()
        return [(row.Entity, row.TranscodedVariantComponent, row.TranscodeProfileComponent) for row in results]

    if session:
        return await _get(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _get(new_session)


async def get_assets_using_profile(
    world: World, profile_entity_id: int, session: Optional[Session] = None
) -> list[Tuple[Entity, TranscodedVariantComponent]]:
    """
    Retrieves all assets that were transcoded using a specific profile.
    Returns a list of tuples: (transcoded_entity, variant_component).
    """

    async def _get(db_session: Session):
        stmt = (
            select(Entity, TranscodedVariantComponent)
            .join(TranscodedVariantComponent, Entity.id == TranscodedVariantComponent.entity_id)
            .where(TranscodedVariantComponent.transcode_profile_entity_id == profile_entity_id)
            .options(
                joinedload(Entity.components_collection)  # Ensure components are loaded for the transcoded entity
            )
        )
        results = (await db_session.execute(stmt)).all()
        return [(row.Entity, row.TranscodedVariantComponent) for row in results]

    if session:
        return await _get(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _get(new_session)


# Example of how SystemStage might be defined (if not already available)
# This is just for context, assuming SystemStage is part of dam.core.stages
try:
    from dam.core.stages import SystemStage
except ImportError:
    from enum import Enum

    class SystemStage(Enum):  # pragma: no cover
        INGESTION = "INGESTION"
        METADATA_EXTRACTION = "METADATA_EXTRACTION"
        # ... other stages
        POST_PROCESSING = "POST_PROCESSING"
        TRANSCODING_QUEUE = "TRANSCODING_QUEUE"  # Example if we had async queue
        EVALUATION = "EVALUATION"
        # ... etc.

    print("Note: Using a mock SystemStage enum for transcode_service.py.")

# --- Transcoding Events ---
# These are now defined in dam.core.events to avoid circular imports.
# @dataclass
# class TranscodeJobRequested(BaseEvent):
#     world_name: str
#     source_entity_id: int
#     profile_id: int
#     priority: int = 100
#     output_parent_dir: Optional[Path] = None

# @dataclass
# class TranscodeJobCompleted(BaseEvent):
#     job_id: int
#     world_name: str
#     source_entity_id: int
#     profile_id: int
#     output_entity_id: int
#     output_file_path: Path

# @dataclass
# class TranscodeJobFailed(BaseEvent):
#     job_id: int
#     world_name: str
#     source_entity_id: int
#     profile_id: int
#     error_message: str

# --- Evaluation Events ---
# Also moved to dam.core.events
# @dataclass
# class StartEvaluationForTranscodedAsset(BaseEvent):
#     world_name: str
#     evaluation_run_id: int
#     transcoded_asset_id: int
