import binascii  # For hex string to bytes conversion
import logging
from typing import Optional

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    FindSimilarImagesQuery,
    WebAssetIngestionRequested,  # Import new event
)
from dam.core.system_params import CurrentWorldConfig, WorldSession  # Import Resource
from dam.core.systems import listens_for
from dam.models import (
    ContentHashMD5Component,
    ContentHashSHA256Component,
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,
    ImagePerceptualAHashComponent,
    ImagePerceptualDHashComponent,
    ImagePerceptualPHashComponent,
    OriginalSourceInfoComponent,
    WebsiteProfileComponent,
    WebSourceComponent,
)
from dam.resources.file_storage_resource import FileStorageResource  # Resource
from dam.services import ecs_service, file_operations

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
    file_storage_resource: FileStorageResource,  # Injected resource
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

    # FileStorageResource (file_storage_resource) now handles storage using its own world_config
    content_hash_sha256, physical_storage_path_suffix = file_storage_resource.store_file(
        file_content, original_filename=original_filename
    )
    content_hash_sha256_bytes = binascii.unhexlify(content_hash_sha256)

    # Use a helper for finding existing entity by hash (can be a local helper or from ecs_service)
    existing_entity = ecs_service.find_entity_by_content_hash(session, content_hash_sha256_bytes, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Linking original source."
        )
        md5_hash_hex = await file_operations.calculate_md5_async(filepath_on_disk)
        md5_hash_bytes = binascii.unhexlify(md5_hash_hex)
        existing_md5_components = ecs_service.get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == md5_hash_bytes for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity=entity, hash_value=md5_hash_bytes)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)
    else:
        created_new_entity = True
        entity = ecs_service.create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for '{original_filename}' (SHA256: {content_hash_sha256[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity=entity, hash_value=content_hash_sha256_bytes)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        md5_hash_hex = await file_operations.calculate_md5_async(filepath_on_disk)
        md5_hash_bytes = binascii.unhexlify(md5_hash_hex)
        chc_md5 = ContentHashMD5Component(entity=entity, hash_value=md5_hash_bytes)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
            entity=entity,
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        ecs_service.add_component_to_entity(session, entity.id, fpc)

        flc = FileLocationComponent(
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
        entity=entity,
        original_filename=original_filename,
        original_path=str(filepath_on_disk.resolve()),
        source_type="local_file",  # Set source_type
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes_hex = await file_operations.generate_perceptual_hashes_async(filepath_on_disk)

        if "phash" in perceptual_hashes_hex:
            phash_bytes = binascii.unhexlify(perceptual_hashes_hex["phash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualPHashComponent, {"hash_value": phash_bytes}
            ):
                iphc = ImagePerceptualPHashComponent(entity=entity, hash_value=phash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, iphc)

        if "ahash" in perceptual_hashes_hex:
            ahash_bytes = binascii.unhexlify(perceptual_hashes_hex["ahash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualAHashComponent, {"hash_value": ahash_bytes}
            ):
                iahc = ImagePerceptualAHashComponent(entity=entity, hash_value=ahash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, iahc)

        if "dhash" in perceptual_hashes_hex:
            dhash_bytes = binascii.unhexlify(perceptual_hashes_hex["dhash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualDHashComponent, {"hash_value": dhash_bytes}
            ):
                idhc = ImagePerceptualDHashComponent(entity=entity, hash_value=dhash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity=entity)
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
        content_hash_sha256_hex = await file_operations.calculate_sha256_async(filepath_on_disk)
        content_hash_md5_hex = await file_operations.calculate_md5_async(filepath_on_disk)
    except IOError:
        logger.exception(f"Error reading file for hashing: {filepath_on_disk} for event {event}")
        return

    content_hash_sha256_bytes = binascii.unhexlify(content_hash_sha256_hex)
    content_hash_md5_bytes = binascii.unhexlify(content_hash_md5_hex)

    existing_entity = ecs_service.find_entity_by_content_hash(session, content_hash_sha256_bytes, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256_hex[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Adding new reference."
        )
        existing_md5_components = ecs_service.get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == content_hash_md5_bytes for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity=entity, hash_value=content_hash_md5_bytes)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)
    else:
        created_new_entity = True
        entity = ecs_service.create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for referenced file '{original_filename}' "
            f"(SHA256: {content_hash_sha256_hex[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity=entity, hash_value=content_hash_sha256_bytes)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        chc_md5 = ContentHashMD5Component(entity=entity, hash_value=content_hash_md5_bytes)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
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
            entity=entity,
            content_identifier=content_hash_sha256_hex,  # Keep hex string for content_identifier if it's for human/external readability
            storage_type="local_reference",
            physical_path_or_key=resolved_original_path,
            contextual_filename=original_filename,
        )
        ecs_service.add_component_to_entity(session, entity.id, flc)

    osi_comp = OriginalSourceInfoComponent(
        entity=entity,
        original_filename=original_filename,
        original_path=resolved_original_path,
        source_type="referenced_file",  # Set source_type
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes_hex = await file_operations.generate_perceptual_hashes_async(filepath_on_disk)

        if "phash" in perceptual_hashes_hex:
            phash_bytes = binascii.unhexlify(perceptual_hashes_hex["phash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualPHashComponent, {"hash_value": phash_bytes}
            ):
                iphc = ImagePerceptualPHashComponent(entity=entity, hash_value=phash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, iphc)

        if "ahash" in perceptual_hashes_hex:
            ahash_bytes = binascii.unhexlify(perceptual_hashes_hex["ahash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualAHashComponent, {"hash_value": ahash_bytes}
            ):
                iahc = ImagePerceptualAHashComponent(entity=entity, hash_value=ahash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, iahc)

        if "dhash" in perceptual_hashes_hex:
            dhash_bytes = binascii.unhexlify(perceptual_hashes_hex["dhash"])
            if not ecs_service.get_components_by_value(
                session, entity.id, ImagePerceptualDHashComponent, {"hash_value": dhash_bytes}
            ):
                idhc = ImagePerceptualDHashComponent(entity=entity, hash_value=dhash_bytes)
                ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity=entity)
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
        f"System handling FindEntityByHashQuery for hash: {event.hash_value} (type: {event.hash_type}) in world {event.world_name} (Req ID: {event.request_id})"
    )
    try:
        hash_bytes = binascii.unhexlify(event.hash_value)
    except binascii.Error as e:
        logger.error(
            f"[QueryResult RequestID: {event.request_id}] Invalid hex string for hash_value '{event.hash_value}': {e}"
        )
        return

    entity = ecs_service.find_entity_by_content_hash(session, hash_bytes, event.hash_type)

    if entity:
        logger.info(
            f"[QueryResult RequestID: {event.request_id}] Found Entity ID: {entity.id} for hash {event.hash_value}"
        )
        # Populate event.result with details
        entity_details = {"entity_id": entity.id, "components": {}}

        # Fetch and add common components
        fpc = ecs_service.get_component(session, entity.id, FilePropertiesComponent)
        if fpc:
            entity_details["components"]["FilePropertiesComponent"] = {
                "original_filename": fpc.original_filename,
                "file_size_bytes": fpc.file_size_bytes,
                "mime_type": fpc.mime_type,
            }

        flcs = ecs_service.get_components(session, entity.id, FileLocationComponent)
        if flcs:
            entity_details["components"]["FileLocationComponent"] = [
                {
                    "content_identifier": flc.content_identifier,
                    "storage_type": flc.storage_type,
                    "physical_path_or_key": flc.physical_path_or_key,
                    "contextual_filename": flc.contextual_filename,
                }
                for flc in flcs
            ]

        sha256_comp = ecs_service.get_component(session, entity.id, ContentHashSHA256Component)
        if sha256_comp:
            entity_details["components"]["ContentHashSHA256Component"] = {"hash_value": sha256_comp.hash_value.hex()}

        md5_comp = ecs_service.get_component(session, entity.id, ContentHashMD5Component)
        if md5_comp:
            entity_details["components"]["ContentHashMD5Component"] = {"hash_value": md5_comp.hash_value.hex()}

        event.result = entity_details
    else:
        logger.info(f"[QueryResult RequestID: {event.request_id}] No entity found for hash {event.hash_value}")
        event.result = None  # Explicitly set to None if not found


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
            source_content_hash_hex = await file_operations.calculate_sha256_async(event.image_path)
            source_content_hash_bytes = binascii.unhexlify(source_content_hash_hex)
            # ecs_service.find_entity_by_content_hash is synchronous
            source_entity = ecs_service.find_entity_by_content_hash(session, source_content_hash_bytes, "sha256")
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
                    # Convert stored bytes hash to hex string for imagehash library
                    db_phash_hex = p_comp.hash_value.hex()
                    db_phash_obj = imagehash.hex_to_hash(db_phash_hex)
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
                    db_ahash_hex = a_comp.hash_value.hex()
                    db_ahash_obj = imagehash.hex_to_hash(db_ahash_hex)
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
                    db_dhash_hex = d_comp.hash_value.hex()
                    db_dhash_obj = imagehash.hex_to_hash(db_dhash_hex)
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
        event.result = similar_entities_info

    except ValueError as ve:
        logger.warning(f"[QueryResult RequestID: {event.request_id}] Error processing image for similarity: {ve}")
        event.result = [{"error": str(ve)}]
    except Exception as e:
        logger.error(
            f"[QueryResult RequestID: {event.request_id}] Unexpected error in similarity search: {e}", exc_info=True
        )
        event.result = [{"error": "Unexpected error during similarity search"}]


# Ensure async versions of file_operations are available or implement them.
# e.g., in file_operations.py:
# async def read_file_async(path): return await asyncio.to_thread(path.read_bytes)
# async def calculate_md5_async(path): return await asyncio.to_thread(calculate_md5, path)
# ...and so on for other file ops.
# For now, assuming they exist or would be added.

__all__ = [
    "handle_asset_file_ingestion_request",
    "handle_asset_reference_ingestion_request",
    "handle_find_entity_by_hash_query",
    "handle_find_similar_images_query",
    "handle_web_asset_ingestion_request",  # Add new handler
]


@listens_for(WebAssetIngestionRequested)
async def handle_web_asset_ingestion_request(
    event: WebAssetIngestionRequested,
    session: WorldSession,
    # world_config: CurrentWorldConfig, # May not be needed directly if not downloading
):
    """
    Handles the ingestion of an asset from a web source.
    Primarily stores metadata and URLs. File download/processing is deferred.
    """
    logger.info(
        f"System handling WebAssetIngestionRequested for URL: {event.source_url} from website: {event.website_identifier_url} in world {event.world_name}"
    )

    # 1. Find or Create Website Entity
    website_entity: Optional[Entity] = None
    # Query for existing WebsiteProfileComponent by main_url
    existing_website_profiles = ecs_service.find_entities_by_component_attribute_value(
        session, WebsiteProfileComponent, "main_url", event.website_identifier_url
    )
    if existing_website_profiles:
        website_entity = existing_website_profiles[0]  # Should be unique by main_url due to model constraint
        logger.info(f"Found existing Website Entity ID {website_entity.id} for URL {event.website_identifier_url}")
    else:
        website_entity = ecs_service.create_entity(session)
        logger.info(f"Creating new Website Entity ID {website_entity.id} for URL {event.website_identifier_url}")

        website_name = event.metadata_payload.get("website_name") if event.metadata_payload else None
        if not website_name:  # Derive from URL if not provided
            try:
                from urllib.parse import urlparse

                parsed_url = urlparse(event.website_identifier_url)
                website_name = parsed_url.netloc.replace("www.", "")
            except Exception:
                website_name = "Unknown Website"

        profile_comp = WebsiteProfileComponent(
            entity_id=website_entity.id,
            entity=website_entity,
            name=website_name,  # Name could come from metadata_payload or be derived
            main_url=event.website_identifier_url,
            description=event.metadata_payload.get("website_description") if event.metadata_payload else None,
            # icon_url, api_endpoint, parser_rules can also be populated from metadata_payload if available
        )
        ecs_service.add_component_to_entity(session, website_entity.id, profile_comp)

    # 2. Create Asset Entity
    asset_entity = ecs_service.create_entity(session)
    logger.info(f"Creating new Asset Entity ID {asset_entity.id} for web asset from URL: {event.source_url}")

    # 3. Create OriginalSourceInfoComponent for the Asset Entity
    # Filename could be derived from URL or title if available
    original_filename = event.metadata_payload.get("asset_title") if event.metadata_payload else None
    if not original_filename:
        try:
            original_filename = event.source_url.split("/")[-1] or f"web_asset_{entity.id}"
        except Exception:
            original_filename = f"web_asset_{entity.id}"

    osi_comp = OriginalSourceInfoComponent(
        entity_id=asset_entity.id,
        entity=asset_entity,
        original_filename=original_filename,
        original_path=event.source_url,  # Store the source URL as the 'path'
        source_type="web_source",  # New source type
    )
    ecs_service.add_component_to_entity(session, asset_entity.id, osi_comp)

    # 4. Create WebSourceComponent for the Asset Entity
    web_source_data = {
        "entity_id": asset_entity.id,
        "entity": asset_entity,
        "website_entity_id": website_entity.id,  # Link to the Website Entity
        "source_url": event.source_url,
        "original_file_url": event.original_file_url,
        # website_name is no longer here
    }
    if event.metadata_payload:
        # web_source_data["website_name"] = event.metadata_payload.get("website_name") # Removed
        web_source_data["gallery_id"] = event.metadata_payload.get("gallery_id")
        web_source_data["uploader_name"] = event.metadata_payload.get("uploader_name")
        web_source_data["uploader_url"] = event.metadata_payload.get("uploader_url")
        web_source_data["asset_title"] = event.metadata_payload.get("asset_title", original_filename)
        web_source_data["asset_description"] = event.metadata_payload.get("asset_description")

        # Handle upload_date conversion if it's a string
        upload_date_str = event.metadata_payload.get("upload_date")
        if upload_date_str:
            try:
                from datetime import datetime

                # Attempt to parse ISO format, add more formats if needed
                web_source_data["upload_date"] = datetime.fromisoformat(upload_date_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Could not parse upload_date string '{upload_date_str}' for {event.source_url}")
                web_source_data["upload_date"] = None
        else:
            web_source_data["upload_date"] = None

        # Store the rest of the payload in raw_metadata_dump
        web_source_data["raw_metadata_dump"] = event.metadata_payload

    if event.tags:
        import json

        web_source_data["tags_json"] = json.dumps(event.tags)

    # Filter out None values for fields that are not explicitly in WebSourceComponent model
    # or rely on the component's default None values.
    # The WebSourceComponent(**web_source_data) will handle this if fields are Optional.
    # However, let's be explicit for required fields or those that might not be in metadata_payload.

    # Clean web_source_data to only include keys that are actual fields in WebSourceComponent
    # to prevent errors if metadata_payload has extra keys not directly mappable.
    # This is less critical if WebSourceComponent uses **kwargs or similar, but good practice.
    # For now, assuming direct field mapping from the keys prepared above.

    web_comp = WebSourceComponent(
        **{k: v for k, v in web_source_data.items() if hasattr(WebSourceComponent, k) and v is not None}
    )
    ecs_service.add_component_to_entity(session, asset_entity.id, web_comp)

    # For web assets ingested this way (metadata-only), we might not immediately
    # add NeedsMetadataExtractionComponent unless we plan to download and process the file.
    # If a file is downloaded later, that process would add the marker.
    # If some metadata *can* be extracted from URLs or existing metadata (e.g., image dimensions from API),
    # a different marker or a direct call to a metadata system could be made.
    # For now, keeping it simple: no file download, no immediate local metadata extraction.

    logger.info(
        f"Finished WebAssetIngestionRequested for Asset Entity ID {asset_entity.id} (Website Entity ID {website_entity.id}) from URL: {event.source_url}"
    )
