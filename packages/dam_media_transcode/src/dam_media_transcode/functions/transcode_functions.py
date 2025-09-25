import asyncio
import uuid
from pathlib import Path
from typing import Optional, Tuple, cast

from dam.core.config import settings
from dam.core.stages import SystemStage
from dam.core.world import World
from dam.functions import (
    ecs_functions,
    tag_functions,
)
from dam.models.core.entity import Entity
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream
from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils import url_utils
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from dam_media_transcode.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam_media_transcode.models.conceptual.transcoded_variant_component import TranscodedVariantComponent
from dam_media_transcode.utils.media_utils import TranscodeError, transcode_media


class TranscodeFunctionsError(Exception):
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
    async with world.transaction_manager() as tx:
        session = tx.session
        # Check if profile with this name already exists
        stmt_existing = select(TranscodeProfileComponent).where(TranscodeProfileComponent.profile_name == profile_name)
        existing_profile = (await session.execute(stmt_existing)).scalars().first()
        if existing_profile:
            raise TranscodeFunctionsError(f"Transcode profile '{profile_name}' already exists.")

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
        await ecs_functions.add_component_to_entity(session, profile_entity.id, profile_component)

        # Add a tag to mark this entity as a "Transcode Profile"
        # This uses the existing tag_functions
        try:
            tag_concept_name = "System:TranscodeProfile"
            tag_concept_entity: Optional[Entity] = None
            # Ensure the tag concept exists
            try:
                tag_concept_entity = await tag_functions.get_tag_concept_by_name(session, tag_concept_name)
            except tag_functions.TagConceptNotFoundError:
                tag_concept_entity = await tag_functions.create_tag_concept(
                    session,
                    tag_name=tag_concept_name,
                    description="Marks an entity as a transcoding profile.",
                    scope_type="GLOBAL",  # Or a more specific scope if desired
                )

            if tag_concept_entity is None:
                raise TranscodeFunctionsError(f"Could not get or create tag concept '{tag_concept_name}'")
            await tag_functions.apply_tag_to_entity(
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
    world: World, profile_identifier: str | int, session: Optional[AsyncSession] = None
) -> Tuple[Entity, TranscodeProfileComponent]:
    """
    Retrieves a transcode profile entity and its component by name or entity ID.
    """

    async def _get(db_session: AsyncSession) -> Tuple[Entity, TranscodeProfileComponent]:
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
            raise TranscodeFunctionsError(f"Transcode profile '{profile_identifier}' not found.")
        return result[0], result[1]  # Entity, TranscodeProfileComponent

    if session:
        return await _get(session)
    else:
        async with world.transaction_manager() as tx:
            return await _get(tx.session)


async def _get_source_asset_filepath(world: World, asset_entity_id: int, session: AsyncSession) -> Path:
    """Helper to get a readable filepath for a source asset using its URL."""
    flc = await ecs_functions.get_component(
        session,
        entity_id=asset_entity_id,
        component_type=FileLocationComponent,
    )
    if not flc or not flc.url:
        raise TranscodeFunctionsError(f"Asset entity {asset_entity_id} has no URL in its FileLocationComponent.")

    try:
        # Use the url_utils to resolve the URL to a local, absolute path
        # This requires the world's config to resolve 'local_cas' URLs
        source_path = url_utils.get_local_path_for_url(flc.url)
    except ValueError as e:
        # Raised by url_utils if URL scheme is not supported for local access
        raise TranscodeFunctionsError(
            f"Could not resolve a local path for asset {asset_entity_id} with URL '{flc.url}': {e}"
        ) from e

    if not source_path or not source_path.exists() or not source_path.is_file():
        raise TranscodeFunctionsError(
            f"Resolved path for asset {asset_entity_id} ('{source_path}') does not exist or is not a file."
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
    async with world.transaction_manager() as tx:
        session = tx.session
        # 1. Get Transcode Profile
        _profile_entity, profile_component = await get_transcode_profile_by_name_or_id(
            world, profile_entity_id, session=session
        )

        # 2. Get Source Asset's File Path
        source_entity = await ecs_functions.get_entity(session, source_asset_entity_id)
        if not source_entity:
            raise TranscodeFunctionsError(f"Source asset entity ID {source_asset_entity_id} not found.")

        source_filepath = await _get_source_asset_filepath(world, source_asset_entity_id, session)

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
            transcoded_filepath = await asyncio.to_thread(
                transcode_media,
                input_path=source_filepath,
                output_path=temp_output_filepath,
                tool_name=profile_component.tool_name,
                tool_params=profile_component.parameters,
            )
        except TranscodeError as e:
            raise TranscodeFunctionsError(f"Transcoding failed: {e}") from e
        except FileNotFoundError as e:  # e.g. input file gone missing
            raise TranscodeFunctionsError(f"Transcoding input file error: {e}") from e

        if not transcoded_filepath.exists() or transcoded_filepath.stat().st_size == 0:
            # Should be caught by transcode_media, but as a safeguard:
            if transcoded_filepath.exists():
                transcoded_filepath.unlink(missing_ok=True)
            raise TranscodeFunctionsError(f"Transcoding produced no output or an empty file at {transcoded_filepath}.")

        # 5. Ingest the Transcoded File as a New Asset
        # This uses the existing asset ingestion event flow.
        # The ingestion system will calculate hashes, extract metadata, and create FileLocationComponent.
        ingestion_command = RegisterLocalFileCommand(file_path=transcoded_filepath)

        # Dispatch command. The handler will ingest the file.
        await world.dispatch_command(ingestion_command).get_all_results()

        # To ensure the file is processed by ingestion systems (metadata, etc.)
        # This would typically run after the event that adds NeedsMetadataExtractionComponent
        await world.execute_stage(SystemStage.METADATA_EXTRACTION)  # type: ignore

        # Find the newly ingested asset by its hash.
        # The ingestion system should have added ContentHashSHA256Component.
        with open(transcoded_filepath, "rb") as f:
            hashes = calculate_hashes_from_stream(f, {HashAlgorithm.SHA256})
        transcoded_file_sha256_bytes = cast(bytes, hashes[HashAlgorithm.SHA256])

        # Query for the entity with this SHA256 hash
        # Use ecs_functions.find_entity_by_content_hash
        newly_ingested_entity_result = await ecs_functions.find_entity_by_content_hash(
            session, transcoded_file_sha256_bytes, "sha256"
        )

        if not newly_ingested_entity_result:
            transcoded_filepath.unlink(missing_ok=True)  # Clean up temp file
            # This could happen if ingestion failed silently or hash mismatch.
            raise TranscodeFunctionsError(
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
        await ecs_functions.add_component_to_entity(session, transcoded_asset_entity.id, transcoded_variant_comp)

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
    world: World, original_asset_entity_id: int, session: Optional[AsyncSession] = None
) -> list[Tuple[Entity, TranscodedVariantComponent, TranscodeProfileComponent]]:
    """
    Retrieves all transcoded variants for a given original asset entity.
    Returns a list of tuples: (transcoded_entity, variant_component, profile_component).
    """

    async def _get(
        db_session: AsyncSession,
    ) -> list[Tuple[Entity, TranscodedVariantComponent, TranscodeProfileComponent]]:
        stmt = (
            select(Entity, TranscodedVariantComponent, TranscodeProfileComponent)
            .join(TranscodedVariantComponent, Entity.id == TranscodedVariantComponent.entity_id)
            .join(
                TranscodeProfileComponent,
                TranscodedVariantComponent.transcode_profile_entity_id == TranscodeProfileComponent.entity_id,
            )
            .where(TranscodedVariantComponent.original_asset_entity_id == original_asset_entity_id)
        )
        results = (await db_session.execute(stmt)).all()
        return [(row.Entity, row.TranscodedVariantComponent, row.TranscodeProfileComponent) for row in results]

    if session:
        return await _get(session)
    else:
        async with world.transaction_manager() as tx:
            return await _get(tx.session)


async def get_assets_using_profile(
    world: World, profile_entity_id: int, session: Optional[AsyncSession] = None
) -> list[Tuple[Entity, TranscodedVariantComponent]]:
    """
    Retrieves all assets that were transcoded using a specific profile.
    Returns a list of tuples: (transcoded_entity, variant_component).
    """

    async def _get(db_session: AsyncSession) -> list[Tuple[Entity, TranscodedVariantComponent]]:
        stmt = (
            select(Entity, TranscodedVariantComponent)
            .join(TranscodedVariantComponent, Entity.id == TranscodedVariantComponent.entity_id)
            .where(TranscodedVariantComponent.transcode_profile_entity_id == profile_entity_id)
        )
        results = (await db_session.execute(stmt)).all()
        return [(row.Entity, row.TranscodedVariantComponent) for row in results]

    if session:
        return await _get(session)
    else:
        async with world.transaction_manager() as tx:
            return await _get(tx.session)
