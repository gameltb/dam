import logging
from typing import Optional

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    FindSimilarImagesQuery,
)
from dam.core.system_params import CurrentWorldConfig, WorldSession  # Import Resource
from dam.core.systems import listens_for
from dam.models import Entity
from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.models.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from dam.models.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent
from dam.models.original_source_info_component import OriginalSourceInfoComponent
from dam.services import ecs_service, file_operations
from dam.services.file_storage_service import FileStorageService  # Resource

# For find_similar_images
try:
    import imagehash
except ImportError:
    imagehash = None


logger = logging.getLogger(__name__)

# --- Command Systems (Event Handlers for Ingestion) ---


@listens_for(AssetFileIngestionRequested)
async def handle_asset_file_ingestion_request(  # Renamed function
    event: AssetFileIngestionRequested,
    session: WorldSession,
    file_storage_svc: FileStorageService,  # Injected resource
    # world_config: CurrentWorldConfig, # Can get from file_storage_svc.world_config
):
    """
    Handles the ingestion of an asset file by copying it, based on an event.
    Logic moved from asset_service.add_asset_file.
    """
    logger.info(
        f"System handling AssetFileIngestionRequested for: {event.original_filename} in world {event.world_name}"
    )
    created_new_entity = False
    filepath_on_disk = event.filepath_on_disk
    original_filename = event.original_filename
    mime_type = event.mime_type
    size_bytes = event.size_bytes

    try:
        file_content = await file_operations.read_file_async(filepath_on_disk)  # Using async read
    except IOError:
        logger.exception(f"Error reading file {filepath_on_disk} for event {event}")
        # Optionally, dispatch a failure event or log more formally
        return  # Stop processing this event

    # FileStorageService (file_storage_svc) now handles storage using its own world_config
    content_hash_sha256, physical_storage_path_suffix = file_storage_svc.store_file(
        file_content, original_filename=original_filename
    )

    # Use a helper for finding existing entity by hash (can be a local helper or from ecs_service)
    existing_entity = ecs_service.find_entity_by_content_hash(session, content_hash_sha256, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Linking original source."
        )
        md5_hash_value = await file_operations.calculate_md5_async(filepath_on_disk)
        existing_md5_components = ecs_service.get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == md5_hash_value for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)
    else:
        created_new_entity = True
        entity = ecs_service.create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for '{original_filename}' (SHA256: {content_hash_sha256[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        md5_hash_value = await file_operations.calculate_md5_async(filepath_on_disk)
        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        ecs_service.add_component_to_entity(session, entity.id, fpc)

        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            content_identifier=content_hash_sha256,
            storage_type="local_cas",
            physical_path_or_key=physical_storage_path_suffix,
            contextual_filename=original_filename,
        )
        ecs_service.add_component_to_entity(session, entity.id, flc)

    if not entity:  # Should not happen if logic is correct
        logger.error(f"Entity object not available after processing {original_filename}. This is unexpected.")
        return

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id,
        entity=entity,
        original_filename=original_filename,
        original_path=str(filepath_on_disk.resolve()),
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes = await file_operations.generate_perceptual_hashes_async(filepath_on_disk)
        if "phash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualPHashComponent, {"hash_value": perceptual_hashes["phash"]}
        ):
            iphc = ImagePerceptualPHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, iphc)
        if "ahash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualAHashComponent, {"hash_value": perceptual_hashes["ahash"]}
        ):
            iahc = ImagePerceptualAHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, iahc)
        if "dhash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualDHashComponent, {"hash_value": perceptual_hashes["dhash"]}
        ):
            idhc = ImagePerceptualDHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        ecs_service.add_component_to_entity(session, entity.id, marker_comp, flush=False)  # Flush managed by scheduler

    logger.info(f"Finished AssetFileIngestionRequested for Entity ID {entity.id}. New entity: {created_new_entity}")
    # The result (entity_id, created_new_entity) is not directly returned to CLI via event.
    # CLI will infer success from lack of error and can query later if needed.


@listens_for(AssetReferenceIngestionRequested)
async def handle_asset_reference_ingestion_request(  # Renamed function
    event: AssetReferenceIngestionRequested,
    session: WorldSession,
    # file_storage_svc is not strictly needed here as no file is written to DAM storage
):
    """
    Handles the ingestion of an asset by reference, based on an event.
    Logic moved from asset_service.add_asset_reference.
    """
    logger.info(
        f"System handling AssetReferenceIngestionRequested for: {event.original_filename} in world {event.world_name}"
    )
    created_new_entity = False
    filepath_on_disk = event.filepath_on_disk
    original_filename = event.original_filename
    mime_type = event.mime_type
    size_bytes = event.size_bytes

    try:
        content_hash_sha256 = await file_operations.calculate_sha256_async(filepath_on_disk)
        content_hash_md5 = await file_operations.calculate_md5_async(filepath_on_disk)
    except IOError:
        logger.exception(f"Error reading file for hashing: {filepath_on_disk} for event {event}")
        return

    existing_entity = ecs_service.find_entity_by_content_hash(session, content_hash_sha256, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Adding new reference."
        )
        existing_md5_components = ecs_service.get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == content_hash_md5 for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=content_hash_md5)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)
    else:
        created_new_entity = True
        entity = ecs_service.create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for referenced file '{original_filename}' "
            f"(SHA256: {content_hash_sha256[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=content_hash_md5)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        ecs_service.add_component_to_entity(session, entity.id, fpc)

    if not entity:
        logger.error(f"Entity object not available after processing reference {original_filename}. This is unexpected.")
        return

    resolved_original_path = str(filepath_on_disk.resolve())
    existing_locations = ecs_service.get_components(session, entity.id, FileLocationComponent)
    found_ref_location = any(
        loc.storage_type == "local_reference" and loc.physical_path_or_key == resolved_original_path
        for loc in existing_locations
    )

    if not found_ref_location:
        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            content_identifier=content_hash_sha256,
            storage_type="local_reference",
            physical_path_or_key=resolved_original_path,
            contextual_filename=original_filename,
        )
        ecs_service.add_component_to_entity(session, entity.id, flc)

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id,
        entity=entity,
        original_filename=original_filename,
        original_path=resolved_original_path,
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes = await file_operations.generate_perceptual_hashes_async(filepath_on_disk)  # async version
        if "phash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualPHashComponent, {"hash_value": perceptual_hashes["phash"]}
        ):
            iphc = ImagePerceptualPHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, iphc)
        # ... (similar for ahash, dhash)
        if "ahash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualAHashComponent, {"hash_value": perceptual_hashes["ahash"]}
        ):
            iahc = ImagePerceptualAHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, iahc)
        if "dhash" in perceptual_hashes and not ecs_service.get_components_by_value(
            session, entity.id, ImagePerceptualDHashComponent, {"hash_value": perceptual_hashes["dhash"]}
        ):
            idhc = ImagePerceptualDHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"]
            )
            ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        ecs_service.add_component_to_entity(session, entity.id, marker_comp, flush=False)  # Flush managed by scheduler

    logger.info(
        f"Finished AssetReferenceIngestionRequested for Entity ID {entity.id}. New entity: {created_new_entity}"
    )


# --- Query Systems (Event Handlers for Queries) ---
# For query systems, returning results to the synchronous CLI is tricky.
# Options:
# 1. System writes result to a temporary resource, CLI polls/retrieves. (Complex)
# 2. System logs result, CLI parses log. (Brittle)
# 3. CLI makes a synchronous call for queries (not event-driven for queries). (Simplest for queries)
# Given user wants event-driven CLI, we'll explore option 1 or a variation.
# For now, these systems will log the result. The CLI will need adaptation.
# A more robust way would be for the World to manage a temporary "query_results" resource
# that systems can write to using event.request_id as a key.


@listens_for(FindEntityByHashQuery)
async def handle_find_entity_by_hash_query(
    event: FindEntityByHashQuery,
    session: WorldSession,
    world_config: CurrentWorldConfig,  # Example of injecting world_config if needed
):
    logger.info(
        f"System handling FindEntityByHashQuery for hash: {event.hash_value} in world {event.world_name} (Req ID: {event.request_id})"
    )
    entity = ecs_service.find_entity_by_content_hash(session, event.hash_value, event.hash_type)

    if entity:
        logger.info(
            f"[QueryResult RequestID: {event.request_id}] Found Entity ID: {entity.id} for hash {event.hash_value}"
        )
        # How to get this back to CLI? For now, CLI might not get direct output.
        # Option: Store in a temporary resource keyed by event.request_id
        # world.add_resource(QueryResult(event.request_id, entity.id), name=f"query_result_{event.request_id}")
    else:
        logger.info(f"[QueryResult RequestID: {event.request_id}] No entity found for hash {event.hash_value}")
        # world.add_resource(QueryResult(event.request_id, None), name=f"query_result_{event.request_id}")


@listens_for(FindSimilarImagesQuery)
async def handle_find_similar_images_query(
    event: FindSimilarImagesQuery,
    session: WorldSession,
):
    logger.info(
        f"System handling FindSimilarImagesQuery for image: {event.image_path} in world {event.world_name} (Req ID: {event.request_id})"
    )

    if not imagehash:
        logger.warning(
            f"[QueryResult RequestID: {event.request_id}] ImageHash library not available. Cannot perform similarity search."
        )
        # world.add_resource(QueryResult(event.request_id, [], error="ImageHash not available"), ...)
        return

    try:
        # This logic is from asset_service.find_entities_by_similar_image_hashes
        # It needs to be adapted to run within an async system.
        # generate_perceptual_hashes might need an async version or to_thread.
        # For now, assume file_operations has async versions or we wrap sync calls.
        input_hashes = await file_operations.generate_perceptual_hashes_async(event.image_path)
        if not input_hashes:
            msg = f"Could not generate perceptual hashes for {event.image_path.name}."
            logger.warning(f"[QueryResult RequestID: {event.request_id}] {msg}")
            # world.add_resource(QueryResult(event.request_id, [], error=msg), ...)
            return

        # The rest of the similarity logic from asset_service.find_entities_by_similar_image_hashes
        # would go here, adapted for async and using injected session.
        input_phash_obj = imagehash.hex_to_hash(input_hashes["phash"]) if "phash" in input_hashes else None
        input_ahash_obj = imagehash.hex_to_hash(input_hashes["ahash"]) if "ahash" in input_hashes else None
        input_dhash_obj = imagehash.hex_to_hash(input_hashes["dhash"]) if "dhash" in input_hashes else None

        source_entity_id = None
        try:
            # Assuming file_operations.calculate_sha256_async exists
            source_content_hash = await file_operations.calculate_sha256_async(event.image_path)
            # ecs_service.find_entity_by_content_hash is synchronous
            source_entity = ecs_service.find_entity_by_content_hash(session, source_content_hash, "sha256")
            if source_entity:
                source_entity_id = source_entity.id
        except Exception as e_src:
            logger.warning(
                f"Could not determine source entity for {event.image_path.name} to exclude from results: {e_src}"
            )

        potential_matches = []
        from sqlalchemy import select as sql_select  # For direct querying if ecs_service helpers are not sufficient

        if input_phash_obj:
            all_phashes_stmt = sql_select(ImagePerceptualPHashComponent)
            db_phashes_components = session.execute(all_phashes_stmt).scalars().all()
            for p_comp in db_phashes_components:
                if source_entity_id and p_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_phash_obj = imagehash.hex_to_hash(p_comp.hash_value)
                    distance = input_phash_obj - db_phash_obj
                    if distance <= event.phash_threshold:
                        entity = session.get(Entity, p_comp.entity_id)
                        if entity:
                            fpc = ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fpc.original_filename if fpc else "N/A",
                                    "match_type": "phash_match",
                                    "distance": distance,
                                    "hash_type": "phash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing pHash for entity {p_comp.entity_id}: {e_cmp}")

        if input_ahash_obj:
            all_ahashes_stmt = sql_select(ImagePerceptualAHashComponent)
            db_ahashes_components = session.execute(all_ahashes_stmt).scalars().all()
            for a_comp in db_ahashes_components:
                if source_entity_id and a_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_ahash_obj = imagehash.hex_to_hash(a_comp.hash_value)
                    distance = input_ahash_obj - db_ahash_obj
                    if distance <= event.ahash_threshold:
                        entity = session.get(Entity, a_comp.entity_id)
                        if entity:
                            fpc = ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fpc.original_filename if fpc else "N/A",
                                    "match_type": "ahash_match",
                                    "distance": distance,
                                    "hash_type": "ahash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing aHash for entity {a_comp.entity_id}: {e_cmp}")

        if input_dhash_obj:
            all_dhashes_stmt = sql_select(ImagePerceptualDHashComponent)
            db_dhashes_components = session.execute(all_dhashes_stmt).scalars().all()
            for d_comp in db_dhashes_components:
                if source_entity_id and d_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_dhash_obj = imagehash.hex_to_hash(d_comp.hash_value)
                    distance = input_dhash_obj - db_dhash_obj
                    if distance <= event.dhash_threshold:
                        entity = session.get(Entity, d_comp.entity_id)
                        if entity:
                            fpc = ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fpc.original_filename if fpc else "N/A",
                                    "match_type": "dhash_match",
                                    "distance": distance,
                                    "hash_type": "dhash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing dHash for entity {d_comp.entity_id}: {e_cmp}")

        final_matches_map = {}
        for match in potential_matches:
            entity_id = match["entity_id"]
            if entity_id not in final_matches_map or match["distance"] < final_matches_map[entity_id]["distance"]:
                final_matches_map[entity_id] = match

        similar_entities_info = list(final_matches_map.values())
        similar_entities_info.sort(key=lambda x: (x["distance"], x["entity_id"]))

        logger.info(
            f"[QueryResult RequestID: {event.request_id}] Found {len(similar_entities_info)} similar images. Results: {similar_entities_info}"
        )
        # How to return to CLI:
        # world_config.query_results_resource.add_result(event.request_id, similar_entities_info)

    except ValueError as ve:
        logger.warning(f"[QueryResult RequestID: {event.request_id}] Error processing image for similarity: {ve}")
        # world_config.query_results_resource.add_result(event.request_id, error=str(ve))
    except Exception as e:
        logger.error(
            f"[QueryResult RequestID: {event.request_id}] Unexpected error in similarity search: {e}", exc_info=True
        )
        # world_config.query_results_resource.add_result(event.request_id, error="Unexpected error")
        # world.add_resource(QueryResult(event.request_id, [], error="Unexpected error"), ...)


# Ensure async versions of file_operations are available or implement them.
# e.g., in file_operations.py:
# async def read_file_async(path): return await asyncio.to_thread(path.read_bytes)
# async def calculate_md5_async(path): return await asyncio.to_thread(calculate_md5, path)
# ...and so on for other file ops.
# For now, assuming they exist or would be added.

__all__ = [
    "handle_asset_file_ingestion_request",
    "handle_asset_reference_ingestion_request",  # Renamed function
    "handle_find_entity_by_hash_query",
    "handle_find_similar_images_query",
]
