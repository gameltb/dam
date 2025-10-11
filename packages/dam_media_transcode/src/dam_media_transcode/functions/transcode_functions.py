"""Provides functions for managing and applying transcoding profiles."""

import asyncio
import logging
import uuid
from pathlib import Path

from dam.core.config import get_dam_toml
from dam.core.stages import SystemStage
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import (
    ecs_functions,
    tag_functions,
)
from dam.models.core.entity import Entity
from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils import url_utils
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..models.conceptual.transcode_profile_component import TranscodeProfileComponent
from ..models.conceptual.transcoded_variant_component import TranscodedVariantComponent
from ..utils.media_utils import TranscodeError, transcode_media

logger = logging.getLogger(__name__)


class TranscodeFunctionsError(Exception):
    """Custom exception for TranscodeService errors."""


async def create_transcode_profile(
    world: World,
    profile_name: str,
    tool_name: str,
    parameters: str,
    output_format: str,
    description: str | None = None,
) -> Entity:
    """Create a new transcoding profile as a conceptual asset."""
    async with world.get_context(WorldTransaction)() as tx:
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
        profile_component = TranscodeProfileComponent(
            id=profile_entity.id,
            profile_name=profile_name,
            tool_name=tool_name,
            parameters=parameters,
            output_format=output_format,
            description=description,
            concept_name=profile_name,
            concept_description=description,
        )
        await ecs_functions.add_component_to_entity(session, profile_entity.id, profile_component)

        # Add a tag to mark this entity as a "Transcode Profile"
        try:
            tag_concept_name = "System:TranscodeProfile"
            tag_concept_entity: Entity | None = None
            try:
                tag_concept_entity = await tag_functions.get_tag_concept_by_name(session, tag_concept_name)
            except tag_functions.TagConceptNotFoundError:
                tag_concept_entity = await tag_functions.create_tag_concept(
                    session,
                    tag_name=tag_concept_name,
                    description="Marks an entity as a transcoding profile.",
                    scope_type="GLOBAL",
                )

            if tag_concept_entity is None:
                raise TranscodeFunctionsError(f"Could not get or create tag concept '{tag_concept_name}'")
            await tag_functions.apply_tag_to_entity(
                session,
                entity_id_to_tag=profile_entity.id,
                tag_concept_entity_id=tag_concept_entity.id,
            )
        except Exception as e:
            world.logger.warning(
                "Could not apply system tag to transcode profile '%s': %s", profile_name, e, exc_info=True
            )

        await session.commit()
        await session.refresh(profile_entity)
        await session.refresh(profile_component)

        logger.info("Transcode profile '%s' (Entity ID: %d) created successfully.", profile_name, profile_entity.id)
        return profile_entity


async def get_transcode_profile_by_name_or_id(
    world: World, profile_identifier: str | int, session: AsyncSession | None = None
) -> tuple[Entity, TranscodeProfileComponent]:
    """Retrieve a transcode profile entity and its component by name or entity ID."""

    async def _get(db_session: AsyncSession) -> tuple[Entity, TranscodeProfileComponent]:
        if isinstance(profile_identifier, int):
            stmt = (
                select(Entity, TranscodeProfileComponent)
                .join(TranscodeProfileComponent, Entity.id == TranscodeProfileComponent.entity_id)
                .where(Entity.id == profile_identifier)
            )
        else:
            stmt = (
                select(Entity, TranscodeProfileComponent)
                .join(TranscodeProfileComponent, Entity.id == TranscodeProfileComponent.entity_id)
                .where(TranscodeProfileComponent.profile_name == profile_identifier)
            )

        result = (await db_session.execute(stmt)).first()
        if not result:
            raise TranscodeFunctionsError(f"Transcode profile '{profile_identifier}' not found.")
        return result[0], result[1]

    if session:
        return await _get(session)
    async with world.get_context(WorldTransaction)() as tx:
        return await _get(tx.session)


async def _get_source_asset_filepath(_world: World, asset_entity_id: int, session: AsyncSession) -> Path:
    """Get a readable filepath for a source asset using its URL."""
    flc = await ecs_functions.get_component(
        session,
        entity_id=asset_entity_id,
        component_type=FileLocationComponent,
    )
    if not flc or not flc.url:
        raise TranscodeFunctionsError(f"Asset entity {asset_entity_id} has no URL in its FileLocationComponent.")

    try:
        source_path = url_utils.get_local_path_for_url(flc.url)
    except ValueError as e:
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
    profile_entity_id: int,
    output_parent_dir: Path | None = None,
) -> Entity:
    """Apply a transcoding profile to a source asset, create a new asset, and link them."""
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        _profile_entity, profile_component = await get_transcode_profile_by_name_or_id(
            world, profile_entity_id, session=session
        )

        source_entity = await ecs_functions.get_entity(session, source_asset_entity_id)
        if not source_entity:
            raise TranscodeFunctionsError(f"Source asset entity ID {source_asset_entity_id} not found.")

        source_filepath = await _get_source_asset_filepath(world, source_asset_entity_id, session)

        temp_transcode_dir = get_dam_toml().get_tool_config().TRANSCODING_TEMP_DIR
        temp_transcode_dir.mkdir(parents=True, exist_ok=True)

        final_output_dir_base = output_parent_dir or temp_transcode_dir
        final_output_dir_base.mkdir(parents=True, exist_ok=True)

        unique_suffix = uuid.uuid4().hex[:8]
        temp_output_filename = f"{Path(source_filepath).stem}_{profile_component.profile_name.replace(' ', '_')}_{unique_suffix}.{profile_component.output_format}"
        temp_output_filepath = final_output_dir_base / temp_output_filename

        logger.info(
            "Applying profile '%s' to asset ID %d (%s)",
            profile_component.profile_name,
            source_asset_entity_id,
            source_filepath,
        )
        logger.info("Output will be temporarily written to: %s", temp_output_filepath)

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
        except FileNotFoundError as e:
            raise TranscodeFunctionsError(f"Transcoding input file error: {e}") from e

        if not transcoded_filepath.exists() or transcoded_filepath.stat().st_size == 0:
            if transcoded_filepath.exists():
                transcoded_filepath.unlink(missing_ok=True)
            raise TranscodeFunctionsError(f"Transcoding produced no output or an empty file at {transcoded_filepath}.")

        ingestion_command = RegisterLocalFileCommand(file_path=transcoded_filepath)
        newly_ingested_entity_id = await world.dispatch_command(ingestion_command).get_one_value()

        if not newly_ingested_entity_id:
            raise TranscodeFunctionsError(
                f"Failed to ingest transcoded asset at {transcoded_filepath}. "
                "RegisterLocalFileCommand did not return an entity ID."
            )

        await world.execute_stage(SystemStage.METADATA_EXTRACTION)  # type: ignore
        transcoded_asset_entity = await ecs_functions.get_entity(session, newly_ingested_entity_id)

        if not transcoded_asset_entity:
            transcoded_filepath.unlink(missing_ok=True)
            raise TranscodeFunctionsError(
                f"Failed to find newly ingested transcoded asset with ID {newly_ingested_entity_id}. "
                "Ingestion might have failed."
            )

        logger.info("Transcoded asset ingested. New Entity ID: %d", transcoded_asset_entity.id)

        transcoded_variant_comp = TranscodedVariantComponent(
            original_asset_entity_id=source_asset_entity_id,
            transcode_profile_entity_id=profile_component.entity_id,
            transcoded_file_size_bytes=transcoded_filepath.stat().st_size,
            quality_metric_vmaf=None,
            quality_metric_ssim=None,
            custom_metrics_json=None,
        )
        await ecs_functions.add_component_to_entity(session, transcoded_asset_entity.id, transcoded_variant_comp)

        await session.commit()
        await session.refresh(transcoded_asset_entity)

        logger.info(
            "TranscodedVariantComponent created and linked for new asset entity ID %d.", transcoded_asset_entity.id
        )

        if transcoded_filepath.exists():
            try:
                transcoded_filepath.unlink()
                logger.info("Cleaned up temporary transcoded file: %s", transcoded_filepath)
            except OSError as e:
                logger.warning("Could not delete temporary transcoded file %s: %s", transcoded_filepath, e)

        return transcoded_asset_entity


async def get_transcoded_variants_for_original(
    world: World, original_asset_entity_id: int, session: AsyncSession | None = None
) -> list[tuple[Entity, TranscodedVariantComponent, TranscodeProfileComponent]]:
    """
    Retrieve all transcoded variants for a given original asset entity.

    Returns a list of tuples: (transcoded_entity, variant_component, profile_component).
    """

    async def _get(
        db_session: AsyncSession,
    ) -> list[tuple[Entity, TranscodedVariantComponent, TranscodeProfileComponent]]:
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
    async with world.get_context(WorldTransaction)() as tx:
        return await _get(tx.session)


async def get_assets_using_profile(
    world: World, profile_entity_id: int, session: AsyncSession | None = None
) -> list[tuple[Entity, TranscodedVariantComponent]]:
    """
    Retrieve all assets that were transcoded using a specific profile.

    Returns a list of tuples: (transcoded_entity, variant_component).
    """

    async def _get(db_session: AsyncSession) -> list[tuple[Entity, TranscodedVariantComponent]]:
        stmt = (
            select(Entity, TranscodedVariantComponent)
            .join(TranscodedVariantComponent, Entity.id == TranscodedVariantComponent.entity_id)
            .where(TranscodedVariantComponent.transcode_profile_entity_id == profile_entity_id)
        )
        results = (await db_session.execute(stmt)).all()
        return [(row.Entity, row.TranscodedVariantComponent) for row in results]

    if session:
        return await _get(session)
    async with world.get_context(WorldTransaction)() as tx:
        return await _get(tx.session)
