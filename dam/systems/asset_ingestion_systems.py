import logging
from pathlib import Path
from typing import Tuple, Optional, Any # Ensure Any and Optional are imported

from sqlalchemy.orm import Session
from dam.core.events import AssetFileIngestionRequested, AssetReferenceIngestionRequested
from dam.core.systems import listens_for
# from dam.core.system_params import WorldContext # Not directly used, session, world_name, config are explicit
from dam.models import Entity
from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.models.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from dam.models.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent
from dam.models.original_source_info_component import OriginalSourceInfoComponent
from dam.core.components_markers import NeedsMetadataExtractionComponent

from dam.services import file_storage, ecs_service
from dam.services.file_operations import calculate_md5, calculate_sha256, generate_perceptual_hashes
from dam.core import config as app_config # For world config access
from dam.core.config import WorldConfig # For type hinting world_config

logger = logging.getLogger(__name__)

# Helper function, adapted from asset_service.find_entity_by_content_hash
def _find_entity_by_content_hash(session: Session, hash_value: str, hash_type: str = "sha256") -> Optional[Entity]:
    normalized_hash_type = hash_type.lower()
    # Assuming ecs_service.select is available or we use sqlalchemy.select directly
    from sqlalchemy import select as sql_select

    if normalized_hash_type == "sha256":
        stmt = (
            sql_select(Entity)
            .join(ContentHashSHA256Component, Entity.id == ContentHashSHA256Component.entity_id)
            .where(ContentHashSHA256Component.hash_value == hash_value)
        )
    elif normalized_hash_type == "md5":
        stmt = (
            sql_select(Entity)
            .join(ContentHashMD5Component, Entity.id == ContentHashMD5Component.entity_id)
            .where(ContentHashMD5Component.hash_value == hash_value)
        )
    else:
        logger.error(f"Unsupported hash type for search: {hash_type}")
        return None
    result = session.execute(stmt).scalar_one_or_none()
    return result


@listens_for(AssetFileIngestionRequested)
async def handle_asset_file_ingestion_request(
    event: AssetFileIngestionRequested,
    session: Session,
    # world_name parameter can be derived from event.world_name
    # world_config parameter can be derived from app_config.settings.get_world_config(event.world_name)
):
    """
    Handles the ingestion of an asset file by copying it.
    Logic derived from asset_service.add_asset_file.
    """
    logger.info(f"Handling AssetFileIngestionRequested for: {event.original_filename} in world {event.world_name}")
    created_new_entity = False
    filepath_on_disk = event.filepath_on_disk
    original_filename = event.original_filename
    mime_type = event.mime_type
    size_bytes = event.size_bytes
    world_name = event.world_name

    try:
        file_content = filepath_on_disk.read_bytes()
    except IOError:
        logger.exception(f"Error reading file {filepath_on_disk} for event {event}")
        raise

    world_config_obj: WorldConfig = app_config.settings.get_world_config(world_name)

    content_hash_sha256, physical_storage_path_suffix = file_storage.store_file(
        file_content, world_config=world_config_obj, original_filename=original_filename
    )

    existing_entity = _find_entity_by_content_hash(session, content_hash_sha256, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Linking original source."
        )
        md5_hash_value = calculate_md5(filepath_on_disk)
        existing_md5_components = ecs_service.get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == md5_hash_value for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)
            logger.info(f"Added MD5 hash '{md5_hash_value}' to existing Entity ID {entity.id}.")

        existing_cas_locations = ecs_service.get_components(session, entity.id, FileLocationComponent)
        found_cas_location = False
        for loc in existing_cas_locations:
            if loc.storage_type == "local_cas" and loc.physical_path_or_key == physical_storage_path_suffix:
                found_cas_location = True
                break
        if not found_cas_location:
            flc = FileLocationComponent(
                entity_id=entity.id, entity=entity, content_identifier=content_hash_sha256,
                storage_type="local_cas", physical_path_or_key=physical_storage_path_suffix,
                contextual_filename=original_filename,
            )
            ecs_service.add_component_to_entity(session, entity.id, flc)
    else:
        created_new_entity = True
        entity = ecs_service.create_entity(session) # type: ignore
        logger.info(
            f"Creating new Entity ID {entity.id} for '{original_filename}' (SHA256: {content_hash_sha256[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        md5_hash_value = calculate_md5(filepath_on_disk)
        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
            entity_id=entity.id, entity=entity, original_filename=original_filename,
            file_size_bytes=size_bytes, mime_type=mime_type,
        )
        ecs_service.add_component_to_entity(session, entity.id, fpc)

        flc = FileLocationComponent(
            entity_id=entity.id, entity=entity, content_identifier=content_hash_sha256,
            storage_type="local_cas", physical_path_or_key=physical_storage_path_suffix,
            contextual_filename=original_filename,
        )
        ecs_service.add_component_to_entity(session, entity.id, flc)

    if not entity:
        logger.error(f"Entity object not available after processing {original_filename}. This is unexpected.")
        raise Exception(f"Entity creation/retrieval failed for {original_filename}")

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id, entity=entity, original_filename=original_filename,
        original_path=str(filepath_on_disk.resolve()),
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes = generate_perceptual_hashes(filepath_on_disk)
        if "phash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualPHashComponent, {"hash_value": perceptual_hashes["phash"]}): #type: ignore
            iphc = ImagePerceptualPHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"])
            ecs_service.add_component_to_entity(session, entity.id, iphc)
        if "ahash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualAHashComponent, {"hash_value": perceptual_hashes["ahash"]}): #type: ignore
            iahc = ImagePerceptualAHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"])
            ecs_service.add_component_to_entity(session, entity.id, iahc)
        if "dhash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualDHashComponent, {"hash_value": perceptual_hashes["dhash"]}): #type: ignore
            idhc = ImagePerceptualDHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"])
            ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        ecs_service.add_component_to_entity(session, entity.id, marker_comp, flush=True)
        logger.info(f"Marked Entity ID {entity.id} with NeedsMetadataExtractionComponent.")

    logger.info(f"Finished handling AssetFileIngestionRequested for Entity ID {entity.id}. New entity: {created_new_entity}")


@listens_for(AssetReferenceIngestionRequested)
async def handle_asset_reference_ingestion_request(
    event: AssetReferenceIngestionRequested,
    session: Session,
):
    """
    Handles the ingestion of an asset by reference.
    Logic derived from asset_service.add_asset_reference.
    """
    logger.info(f"Handling AssetReferenceIngestionRequested for: {event.original_filename} in world {event.world_name}")
    created_new_entity = False
    filepath_on_disk = event.filepath_on_disk
    original_filename = event.original_filename
    mime_type = event.mime_type
    size_bytes = event.size_bytes

    try:
        content_hash_sha256 = calculate_sha256(filepath_on_disk)
        content_hash_md5 = calculate_md5(filepath_on_disk)
    except IOError:
        logger.exception(f"Error reading file for hashing: {filepath_on_disk} for event {event}")
        raise

    existing_entity = _find_entity_by_content_hash(session, content_hash_sha256, "sha256")
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
        entity = ecs_service.create_entity(session) # type: ignore
        logger.info(
            f"Creating new Entity ID {entity.id} for referenced file '{original_filename}' "
            f"(SHA256: {content_hash_sha256[:12]}...)."
        )
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=content_hash_md5)
        ecs_service.add_component_to_entity(session, entity.id, chc_md5)

        fpc = FilePropertiesComponent(
            entity_id=entity.id, entity=entity, original_filename=original_filename,
            file_size_bytes=size_bytes, mime_type=mime_type,
        )
        ecs_service.add_component_to_entity(session, entity.id, fpc)

    if not entity:
        logger.error(f"Entity object not available after processing reference {original_filename}. This is unexpected.")
        raise Exception(f"Entity creation/retrieval failed for reference {original_filename}")

    resolved_original_path = str(filepath_on_disk.resolve())

    existing_locations = ecs_service.get_components(session, entity.id, FileLocationComponent)
    found_ref_location = any(
        loc.storage_type == "local_reference" and loc.physical_path_or_key == resolved_original_path
        for loc in existing_locations
    )

    if not found_ref_location:
        flc = FileLocationComponent(
            entity_id=entity.id, entity=entity, content_identifier=content_hash_sha256,
            storage_type="local_reference", physical_path_or_key=resolved_original_path,
            contextual_filename=original_filename,
        )
        ecs_service.add_component_to_entity(session, entity.id, flc)
        logger.info(f"Added new FileLocationComponent (local_reference) for path '{resolved_original_path}' to Entity ID {entity.id}.")

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id, entity=entity, original_filename=original_filename,
        original_path=resolved_original_path,
    )
    ecs_service.add_component_to_entity(session, entity.id, osi_comp)

    if mime_type and mime_type.startswith("image/"):
        perceptual_hashes = generate_perceptual_hashes(filepath_on_disk)
        if "phash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualPHashComponent, {"hash_value": perceptual_hashes["phash"]}): #type: ignore
            iphc = ImagePerceptualPHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"])
            ecs_service.add_component_to_entity(session, entity.id, iphc)
        if "ahash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualAHashComponent, {"hash_value": perceptual_hashes["ahash"]}): #type: ignore
            iahc = ImagePerceptualAHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"])
            ecs_service.add_component_to_entity(session, entity.id, iahc)
        if "dhash" in perceptual_hashes and not ecs_service.get_components_by_value(session, entity.id, ImagePerceptualDHashComponent, {"hash_value": perceptual_hashes["dhash"]}): #type: ignore
            idhc = ImagePerceptualDHashComponent(entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"])
            ecs_service.add_component_to_entity(session, entity.id, idhc)

    if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        ecs_service.add_component_to_entity(session, entity.id, marker_comp, flush=True)
        logger.info(f"Marked Entity ID {entity.id} with NeedsMetadataExtractionComponent for referenced asset.")

    logger.info(f"Finished handling AssetReferenceIngestionRequested for Entity ID {entity.id}. New entity: {created_new_entity}")

__all__ = [
    "handle_asset_file_ingestion_request",
    "handle_asset_reference_ingestion_request",
]
