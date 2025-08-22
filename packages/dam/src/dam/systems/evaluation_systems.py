import json
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.future import select
from sqlalchemy.orm import Session

from dam.core.exceptions import EntityNotFoundError
from dam.core.world import World
from dam.models.conceptual.evaluation_result_component import EvaluationResultComponent
from dam.models.conceptual.evaluation_run_component import EvaluationRunComponent
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.core.entity import Entity
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.services import ecs_service, tag_service, transcode_service


class EvaluationError(Exception):
    """Custom exception for evaluation system errors."""

    pass


# --- Event Definitions (Optional - could directly call service methods too) ---
# class StartEvaluationRun(Event):
#     evaluation_run_name: str
#     source_asset_identifiers: List[str | int] # IDs or hashes
#     profile_identifiers: List[str | int] # IDs or names
#     description: Optional[str] = None

# class EvaluationRunCompleted(Event):
#     evaluation_run_entity_id: int
#     results: List[Dict[str, Any]] # Summary of results


# --- Service-like functions for Evaluation ---


async def create_evaluation_run_concept(
    world: World, run_name: str, description: Optional[str] = None, session: Optional[Session] = None
) -> Entity:
    """
    Creates a new EvaluationRun conceptual asset.
    """

    async def _create(db_session: Session):
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
            id=run_entity.id,  # For EvaluationRunComponent's own PK/FK 'id' field
            run_name=run_name,
            concept_name=run_name,  # Pass run_name for concept_name
            concept_description=description,  # Pass description for concept_description
        )
        # Use ecs_service to add the component and handle associations for BaseComponent fields
        await ecs_service.add_component_to_entity(db_session, run_entity.id, eval_run_comp)

        # Tag it as an "Evaluation Run"
        try:
            tag_concept_name = "System:EvaluationRun"
            try:
                # Pass db_session directly, world is not used by get_tag_concept_by_name
                tag_concept = await tag_service.get_tag_concept_by_name(db_session, tag_concept_name)
            except tag_service.TagConceptNotFoundError:
                # create_tag_concept expects session as its first argument
                tag_concept = await tag_service.create_tag_concept(
                    db_session,  # Pass db_session here
                    tag_name=tag_concept_name,  # Parameter name is tag_name
                    description="Marks an entity as an Evaluation Run.",
                    scope_type="GLOBAL",  # Parameter name is scope_type
                    # session=db_session # Removed redundant keyword argument
                )
            # apply_tag_to_entity call confirmed to be correct
            await tag_service.apply_tag_to_entity(db_session, run_entity.id, tag_concept.id)
        except Exception as e:
            world.logger.warning(f"Could not apply system tag to evaluation run '{run_name}': {e}")

        await db_session.commit()
        await db_session.refresh(run_entity)
        await db_session.refresh(eval_run_comp)
        world.logger.info(f"Evaluation Run '{run_name}' (Entity ID: {run_entity.id}) created.")
        return run_entity

    if session:
        return await _create(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _create(new_session)


async def get_evaluation_run_by_name_or_id(
    world: World, identifier: str | int, session: Optional[Session] = None
) -> Tuple[Entity, EvaluationRunComponent]:
    """Retrieves an evaluation run by its name or entity ID."""

    async def _get(db_session: Session):
        if isinstance(identifier, int):  # Entity ID
            stmt = (
                select(Entity, EvaluationRunComponent)
                .join(EvaluationRunComponent, Entity.id == EvaluationRunComponent.entity_id)
                .where(Entity.id == identifier)
            )
        else:  # Run name
            stmt = (
                select(Entity, EvaluationRunComponent)
                .join(EvaluationRunComponent, Entity.id == EvaluationRunComponent.entity_id)
                .where(EvaluationRunComponent.run_name == identifier)
            )
        result = (await db_session.execute(stmt)).first()
        if not result:
            raise EvaluationError(f"Evaluation run '{identifier}' not found.")
        return result[0], result[1]  # Entity, EvaluationRunComponent

    if session:
        return await _get(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _get(new_session)


async def evaluate_transcode_output(event, world, session):
    # Placeholder for old test, actual logic might be in execute_evaluation_run
    # or an event listener for StartEvaluationForTranscodedAsset
    world.logger.warning("Placeholder evaluate_transcode_output called by an outdated test.")
    pass


async def execute_evaluation_run(
    world: World,
    evaluation_run_id_or_name: str | int,
    source_asset_identifiers: List[str | int],  # Entity IDs or SHA256 Hashes
    profile_identifiers: List[str | int],  # Transcode Profile Entity IDs or Names
    # quality_calculation_fn: Optional[Callable[[Path, Path], Dict[str, Any]]] = None # Path_original, Path_transcoded
) -> List[EvaluationResultComponent]:
    """
    Executes an evaluation run:
    1. Retrieves the EvaluationRunComponent.
    2. For each source asset and each transcode profile:
        a. Calls transcode_service.apply_transcode_profile().
        b. (Future) Calculates quality metrics using quality_calculation_fn.
        c. Creates an EvaluationResultComponent for the transcoded asset.
    Returns a list of created EvaluationResultComponent instances.
    """
    async with world.db_session_maker() as session:
        try:
            _eval_run_entity, eval_run_comp = await get_evaluation_run_by_name_or_id(
                world, evaluation_run_id_or_name, session=session
            )
        except EvaluationError as e:
            world.logger.error(f"Failed to start evaluation: {e}")
            raise

        world.logger.info(f"Starting evaluation run: '{eval_run_comp.run_name}' (Entity ID: {eval_run_comp.entity_id})")

        results: List[EvaluationResultComponent] = []

        # Resolve source asset entity IDs
        source_asset_entity_ids: List[int] = []
        for asset_id_or_hash in source_asset_identifiers:
            if isinstance(asset_id_or_hash, int):
                source_asset_entity_ids.append(asset_id_or_hash)
            else:  # Assume SHA256 hash
                try:
                    entity_id = await ecs_service.find_entity_id_by_hash(session, asset_id_or_hash, "sha256")
                    if entity_id is None:
                        raise EntityNotFoundError(f"Asset with hash {asset_id_or_hash} not found.")
                    source_asset_entity_ids.append(entity_id)
                except EntityNotFoundError as e:
                    world.logger.warning(f"Skipping asset '{asset_id_or_hash}': {e}")
                    continue

        if not source_asset_entity_ids:
            raise EvaluationError("No valid source assets found for evaluation.")

        # Resolve transcode profile entity IDs and components
        profiles_to_run: List[Tuple[Entity, TranscodeProfileComponent]] = []
        for prof_id_or_name in profile_identifiers:
            try:
                profile_entity, profile_comp = await transcode_service.get_transcode_profile_by_name_or_id(
                    world, prof_id_or_name, session=session
                )
                profiles_to_run.append((profile_entity, profile_comp))
            except transcode_service.TranscodeServiceError as e:
                world.logger.warning(f"Skipping profile '{prof_id_or_name}': {e}")
                continue

        if not profiles_to_run:
            raise EvaluationError("No valid transcode profiles found for evaluation.")

        world.logger.info(f"Source Assets for Evaluation ({len(source_asset_entity_ids)}): {source_asset_entity_ids}")
        world.logger.info(
            f"Transcode Profiles for Evaluation ({len(profiles_to_run)}): {[p[1].profile_name for p in profiles_to_run]}"
        )

        for original_asset_entity_id in source_asset_entity_ids:
            world.logger.info(f"Processing original asset ID: {original_asset_entity_id}")
            # Validate original asset exists
            original_asset_entity = await ecs_service.get_entity(
                session, original_asset_entity_id
            )  # Changed to get_entity
            if not original_asset_entity:
                world.logger.warning(f"Original asset ID {original_asset_entity_id} not found. Skipping.")
                continue

            for profile_entity, profile_comp in profiles_to_run:
                world.logger.info(f"  Applying profile: '{profile_comp.profile_name}' (ID: {profile_entity.id})")
                transcoded_asset_entity: Optional[Entity] = None
                try:
                    # apply_transcode_profile manages its own session internally for the core transcoding and ingestion logic.
                    # This is important because it dispatches events that run in separate transaction contexts.
                    transcoded_asset_entity_id_from_service = (
                        await transcode_service.apply_transcode_profile(
                            world=world,
                            source_asset_entity_id=original_asset_entity_id,
                            profile_entity_id=profile_entity.id,
                        )
                    ).id  # Get the ID from the returned entity

                    # After apply_transcode_profile completes and commits its transaction for the new asset,
                    # we fetch the new entity within *this* evaluation run's session to add the EvaluationResultComponent.
                    transcoded_asset_entity = await ecs_service.get_entity(
                        session, transcoded_asset_entity_id_from_service
                    )  # Changed to get_entity
                    if not transcoded_asset_entity:
                        world.logger.error(
                            f"    Failed to retrieve newly transcoded asset ID {transcoded_asset_entity_id_from_service} in current session. Skipping result component."
                        )
                        continue

                    world.logger.info(f"    Successfully transcoded. New asset ID: {transcoded_asset_entity.id}")

                    # await session.refresh(transcoded_asset_entity, attribute_names=['components_collection']) # Keep refresh if entity object is used
                    # Commented out: In mocked scenarios, transcoded_asset_entity is a MagicMock and not in the session.
                    # For real scenarios, if components_collection needs to be eagerly loaded or refreshed,
                    # this would be necessary. However, current code below only fetches FPC by ID.

                    # Use ecs_service.get_component with entity_id
                    fpc = await ecs_service.get_component(
                        session,
                        transcoded_asset_entity.id,
                        FilePropertiesComponent,  # type: ignore
                    )
                    file_size = fpc.file_size_bytes if fpc else None  # type: ignore

                    vmaf = None
                    ssim = None
                    psnr = None
                    custom_metrics = None

                    eval_result_comp = EvaluationResultComponent(
                        evaluation_run_entity_id=eval_run_comp.entity_id,
                        original_asset_entity_id=original_asset_entity_id,
                        transcode_profile_entity_id=profile_entity.id,
                        transcoded_asset_entity_id=transcoded_asset_entity.id,
                        file_size_bytes=file_size,
                        vmaf_score=vmaf,
                        ssim_score=ssim,
                        psnr_score=psnr,
                        custom_metrics_json=custom_metrics,
                        notes=None,  # Add missing 'notes' argument
                    )
                    eval_result_comp.entity_id = transcoded_asset_entity.id  # Set entity_id directly
                    session.add(eval_result_comp)
                    await session.flush()
                    results.append(eval_result_comp)
                    world.logger.info(
                        f"    EvaluationResultComponent created (ID: {eval_result_comp.id}) for asset {transcoded_asset_entity.id}"
                    )

                except transcode_service.TranscodeServiceError as tse:
                    world.logger.error(
                        f"    Failed to transcode asset ID {original_asset_entity_id} with profile '{profile_comp.profile_name}': {tse}"
                    )
                except Exception as e:
                    world.logger.error(
                        f"    Unexpected error processing asset ID {original_asset_entity_id} with profile '{profile_comp.profile_name}': {e}",
                        exc_info=True,
                    )

        await session.commit()  # Commit all EvaluationResultComponents for this run
        world.logger.info(f"Evaluation run '{eval_run_comp.run_name}' completed. {len(results)} results generated.")
        return results


async def get_evaluation_results(
    world: World, evaluation_run_id_or_name: str | int, session: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves and formats results for a given evaluation run.
    """

    async def _get(db_session: Session):
        _eval_run_entity, eval_run_comp = await get_evaluation_run_by_name_or_id(
            world, evaluation_run_id_or_name, session=db_session
        )

        stmt = (
            select(
                EvaluationResultComponent,
                TranscodeProfileComponent,
                Entity,
            )
            .join(
                TranscodeProfileComponent,
                EvaluationResultComponent.transcode_profile_entity_id == TranscodeProfileComponent.entity_id,
            )
            .join(
                Entity, EvaluationResultComponent.entity_id == Entity.id
            )  # This joins EvaluationResultComponent.entity_id (which is the transcoded asset's entity_id) to Entity.id
            .where(EvaluationResultComponent.evaluation_run_entity_id == eval_run_comp.entity_id)
            # Removed problematic .options() for joinedload(Entity.components_collection)
        )

        db_results = (await db_session.execute(stmt)).all()

        formatted_results = []
        for res_comp, prof_comp, transcoded_entity_obj in db_results:  # type: ignore
            # Fetch FilePropertiesComponent for the original asset
            orig_fpc = await ecs_service.get_component(
                db_session, res_comp.original_asset_entity_id, FilePropertiesComponent
            )  # type: ignore
            original_filename = orig_fpc.original_filename if orig_fpc else "N/A"  # type: ignore

            # Fetch FilePropertiesComponent for the transcoded asset
            trans_fpc = await ecs_service.get_component(db_session, transcoded_entity_obj.id, FilePropertiesComponent)  # type: ignore
            transcoded_filename = trans_fpc.original_filename if trans_fpc else "N/A"

            # TranscodedVariantComponent is not directly used in the loop for constructing formatted_results,
            # so no need to fetch it separately unless its fields were to be added to the report.

            custom_metrics = {}
            if res_comp.custom_metrics_json:
                try:
                    custom_metrics = json.loads(res_comp.custom_metrics_json)
                except json.JSONDecodeError:
                    custom_metrics = {"error": "Failed to parse custom_metrics_json"}

            formatted_results.append(
                {
                    "evaluation_result_id": res_comp.id,
                    "evaluation_run_name": eval_run_comp.run_name,
                    "original_asset_entity_id": res_comp.original_asset_entity_id,
                    "original_asset_filename": original_filename,
                    "transcoded_asset_entity_id": res_comp.transcoded_asset_entity_id,  # This is same as transcoded_entity_obj.id
                    "transcoded_asset_filename": transcoded_filename,
                    "profile_name": prof_comp.profile_name,
                    "profile_tool": prof_comp.tool_name,
                    "profile_params": prof_comp.parameters,
                    "profile_format": prof_comp.output_format,
                    "file_size_bytes": res_comp.file_size_bytes,
                    "vmaf_score": res_comp.vmaf_score,
                    "ssim_score": res_comp.ssim_score,
                    "psnr_score": res_comp.psnr_score,
                    "custom_metrics": custom_metrics,
                    "notes": res_comp.notes,
                }
            )
        return formatted_results

    if session:
        return await _get(session)
    else:
        async with world.db_session_maker() as new_session:
            return await _get(new_session)
