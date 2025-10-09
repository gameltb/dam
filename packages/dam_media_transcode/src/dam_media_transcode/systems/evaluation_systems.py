"""Defines systems for running and managing transcoding evaluations."""
import json
from typing import Any

from dam.core.exceptions import EntityNotFoundError
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import ecs_functions, tag_functions
from dam.models.core.entity import Entity
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_fs.models.filename_component import FilenameComponent
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..functions import transcode_functions
from ..models.conceptual.evaluation_result_component import EvaluationResultComponent
from ..models.conceptual.evaluation_run_component import EvaluationRunComponent
from ..models.conceptual.transcode_profile_component import TranscodeProfileComponent


class EvaluationError(Exception):
    """Custom exception for evaluation system errors."""


async def create_evaluation_run_concept(
    world: World, run_name: str, description: str | None = None, session: AsyncSession | None = None
) -> Entity:
    """Create a new EvaluationRun conceptual asset."""

    async def _create(db_session: AsyncSession) -> Entity:
        existing_run_stmt = select(EvaluationRunComponent).where(EvaluationRunComponent.run_name == run_name)
        existing_run = (await db_session.execute(existing_run_stmt)).scalars().first()
        if existing_run:
            raise EvaluationError(
                f"Evaluation run with name '{run_name}' already exists (Entity ID: {existing_run.entity_id})."
            )

        run_entity = Entity()
        db_session.add(run_entity)
        await db_session.flush()

        eval_run_comp = EvaluationRunComponent(
            id=run_entity.id,
            run_name=run_name,
            concept_name=run_name,
            concept_description=description,
        )
        await ecs_functions.add_component_to_entity(db_session, run_entity.id, eval_run_comp)

        try:
            tag_concept_name = "System:EvaluationRun"
            tag_concept = await tag_functions.get_or_create_tag_concept(
                db_session,
                tag_name=tag_concept_name,
                description="Marks an entity as an Evaluation Run.",
                scope_type="GLOBAL",
            )
            await tag_functions.apply_tag_to_entity(db_session, run_entity.id, tag_concept.id)
        except Exception as e:
            world.logger.warning("Could not apply system tag to evaluation run '%s': %s", run_name, e)

        await db_session.commit()
        await db_session.refresh(run_entity)
        await db_session.refresh(eval_run_comp)
        world.logger.info("Evaluation Run '%s' (Entity ID: %d) created.", run_name, run_entity.id)
        return run_entity

    if session:
        return await _create(session)
    async with world.get_context(WorldTransaction)() as tx:
        return await _create(tx.session)


async def get_evaluation_run_by_name_or_id(
    world: World, identifier: str | int, session: AsyncSession | None = None
) -> tuple[Entity, EvaluationRunComponent]:
    """Retrieve an evaluation run by its name or entity ID."""

    async def _get(db_session: AsyncSession) -> tuple[Entity, EvaluationRunComponent]:
        if isinstance(identifier, int):
            stmt = (
                select(Entity, EvaluationRunComponent)
                .join(EvaluationRunComponent, Entity.id == EvaluationRunComponent.entity_id)
                .where(Entity.id == identifier)
            )
        else:
            stmt = (
                select(Entity, EvaluationRunComponent)
                .join(EvaluationRunComponent, Entity.id == EvaluationRunComponent.entity_id)
                .where(EvaluationRunComponent.run_name == identifier)
            )
        result = (await db_session.execute(stmt)).first()
        if not result:
            raise EvaluationError(f"Evaluation run '{identifier}' not found.")
        return result[0], result[1]

    if session:
        return await _get(session)
    async with world.get_context(WorldTransaction)() as tx:
        return await _get(tx.session)


async def evaluate_transcode_output(_event: Any, world: World, _session: Any) -> None:
    """Handle the evaluation of a transcoded output (placeholder)."""
    world.logger.warning("Placeholder evaluate_transcode_output called by an outdated test.")


async def _resolve_source_assets(
    world: World, session: AsyncSession, identifiers: list[str | int]
) -> list[int]:
    """Resolve a list of identifiers to source asset entity IDs."""
    entity_ids = []
    for asset_id_or_hash in identifiers:
        if isinstance(asset_id_or_hash, int):
            entity_ids.append(asset_id_or_hash)
        else:
            try:
                entity_id = await ecs_functions.find_entity_id_by_hash(session, asset_id_or_hash, "sha256")
                if entity_id is None:
                    raise EntityNotFoundError(f"Asset with hash {asset_id_or_hash} not found.")
                entity_ids.append(entity_id)
            except EntityNotFoundError as e:
                world.logger.warning("Skipping asset '%s': %s", asset_id_or_hash, e)
                continue
    if not entity_ids:
        raise EvaluationError("No valid source assets found for evaluation.")
    return entity_ids


async def _resolve_transcode_profiles(
    world: World, session: AsyncSession, identifiers: list[str | int]
) -> list[tuple[Entity, TranscodeProfileComponent]]:
    """Resolve a list of identifiers to transcode profile entities and components."""
    profiles = []
    for prof_id_or_name in identifiers:
        try:
            profile = await transcode_functions.get_transcode_profile_by_name_or_id(
                world, prof_id_or_name, session=session
            )
            profiles.append(profile)
        except transcode_functions.TranscodeFunctionsError as e:
            world.logger.warning("Skipping profile '%s': %s", prof_id_or_name, e)
            continue
    if not profiles:
        raise EvaluationError("No valid transcode profiles found for evaluation.")
    return profiles


async def _process_job(
    world: World, session: AsyncSession, original_asset_id: int, profile_entity: Entity, profile_comp: TranscodeProfileComponent
) -> EvaluationResultComponent | None:
    """Process a single transcoding job within an evaluation run."""
    world.logger.info("  Applying profile: '%s' (ID: %d)", profile_comp.profile_name, profile_entity.id)
    try:
        transcoded_entity = await transcode_functions.apply_transcode_profile(
            world=world,
            source_asset_entity_id=original_asset_id,
            profile_entity_id=profile_entity.id,
        )
        world.logger.info("    Successfully transcoded. New asset ID: %d", transcoded_entity.id)

        clc = await ecs_functions.get_component(session, transcoded_entity.id, ContentLengthComponent)
        eval_result = EvaluationResultComponent(
            evaluation_run_entity_id=(await get_evaluation_run_by_name_or_id(world, world.name))[1].entity_id,
            original_asset_entity_id=original_asset_id,
            transcode_profile_entity_id=profile_entity.id,
            transcoded_asset_entity_id=transcoded_entity.id,
            file_size_bytes=clc.file_size_bytes if clc else None,
        )
        eval_result.entity_id = transcoded_entity.id
        session.add(eval_result)
        await session.flush()
        world.logger.info("    EvaluationResultComponent created (ID: %d) for asset %d", eval_result.id, transcoded_entity.id)
        return eval_result
    except transcode_functions.TranscodeFunctionsError as e:
        world.logger.error(
            "    Failed to transcode asset ID %d with profile '%s': %s",
            original_asset_id,
            profile_comp.profile_name,
            e,
        )
    except Exception:
        world.logger.exception(
            "    Unexpected error processing asset ID %d with profile '%s'",
            original_asset_id,
            profile_comp.profile_name,
        )
    return None


async def execute_evaluation_run(
    world: World,
    evaluation_run_id_or_name: str | int,
    source_asset_identifiers: list[str | int],
    profile_identifiers: list[str | int],
) -> list[EvaluationResultComponent]:
    """
    Execute an evaluation run.

    1. Retrieves the EvaluationRunComponent.
    2. For each source asset and each transcode profile:
        a. Calls transcode_functions.apply_transcode_profile().
        b. Creates an EvaluationResultComponent for the transcoded asset.
    Returns a list of created EvaluationResultComponent instances.
    """
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        try:
            _eval_run_entity, eval_run_comp = await get_evaluation_run_by_name_or_id(
                world, evaluation_run_id_or_name, session=session
            )
        except EvaluationError as e:
            world.logger.error("Failed to start evaluation: %s", e)
            raise

        world.logger.info("Starting evaluation run: '%s' (Entity ID: %d)", eval_run_comp.run_name, eval_run_comp.entity_id)
        source_ids = await _resolve_source_assets(world, session, source_asset_identifiers)
        profiles = await _resolve_transcode_profiles(world, session, profile_identifiers)
        world.logger.info("Source Assets for Evaluation (%d): %s", len(source_ids), source_ids)
        world.logger.info(
            "Transcode Profiles for Evaluation (%d): %s", len(profiles), [p[1].profile_name for p in profiles]
        )

        results = []
        for original_id in source_ids:
            world.logger.info("Processing original asset ID: %d", original_id)
            for profile_entity, profile_comp in profiles:
                result = await _process_job(world, session, original_id, profile_entity, profile_comp)
                if result:
                    results.append(result)

        await session.commit()
        world.logger.info(
            "Evaluation run '%s' completed. %d results generated.", eval_run_comp.run_name, len(results)
        )
        return results


async def get_evaluation_results(
    world: World, evaluation_run_id_or_name: str | int, session: AsyncSession | None = None
) -> list[dict[str, Any]]:
    """Retrieve and format results for a given evaluation run."""

    async def _get(db_session: AsyncSession) -> list[dict[str, Any]]:
        _eval_run_entity, eval_run_comp = await get_evaluation_run_by_name_or_id(
            world, evaluation_run_id_or_name, session=db_session
        )
        stmt = (
            select(EvaluationResultComponent, TranscodeProfileComponent, Entity)
            .join(TranscodeProfileComponent, EvaluationResultComponent.transcode_profile_entity_id == TranscodeProfileComponent.entity_id)
            .join(Entity, EvaluationResultComponent.entity_id == Entity.id)
            .where(EvaluationResultComponent.evaluation_run_entity_id == eval_run_comp.entity_id)
        )
        db_results = (await db_session.execute(stmt)).all()

        formatted = []
        for res_comp, prof_comp, transcoded_entity in db_results:
            orig_fnc = await ecs_functions.get_component(db_session, res_comp.original_asset_entity_id, FilenameComponent)
            trans_fnc = await ecs_functions.get_component(db_session, transcoded_entity.id, FilenameComponent)
            custom_metrics = json.loads(res_comp.custom_metrics_json) if res_comp.custom_metrics_json else {}
            formatted.append({
                "evaluation_result_id": res_comp.id,
                "evaluation_run_name": eval_run_comp.run_name,
                "original_asset_entity_id": res_comp.original_asset_entity_id,
                "original_asset_filename": orig_fnc.filename if orig_fnc else "N/A",
                "transcoded_asset_entity_id": res_comp.transcoded_asset_entity_id,
                "transcoded_asset_filename": trans_fnc.filename if trans_fnc else "N/A",
                "profile_name": prof_comp.profile_name,
                "file_size_bytes": res_comp.file_size_bytes,
                "vmaf_score": res_comp.vmaf_score,
                "ssim_score": res_comp.ssim_score,
                "psnr_score": res_comp.psnr_score,
                "custom_metrics": custom_metrics,
                "notes": res_comp.notes,
            })
        return formatted

    if session:
        return await _get(session)
    async with world.get_context(WorldTransaction)() as tx:
        return await _get(tx.session)
