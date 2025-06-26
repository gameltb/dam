import logging
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.models import Entity
from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.models.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from dam.models.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent

# Conditional import for imagehash
try:
    import imagehash
except ImportError:
    imagehash = None

# system_service is no longer directly called by asset_service for processing.
# Instead, asset_service will add a marker component.
# from . import system_service # No longer needed here directly
from dam.core.components_markers import NeedsMetadataExtractionComponent  # Import marker

from . import file_storage
from .ecs_service import (
    add_component_to_entity,
    create_entity,
    get_components,
)
from .file_operations import (
    calculate_md5,  # Added for MD5 calculation
    calculate_sha256,  # Added for SHA256 calculation for source image
    # store_file_locally, # Replaced by file_storage service
    generate_perceptual_hashes,
)

logger = logging.getLogger(__name__)  # Initialize logger at module level


def find_entity_by_content_hash(session: Session, hash_value: str, hash_type: str = "sha256") -> Optional[Entity]:
    """
    Finds an entity by its content hash (SHA256 or MD5).

    Args:
        session: SQLAlchemy session.
        hash_value: The hash value to search for.
        hash_type: The type of hash (e.g., "sha256").

    Returns:
        The Entity if found, otherwise None.
    """
    normalized_hash_type = hash_type.lower()
    if normalized_hash_type == "sha256":
        stmt = (
            select(Entity)
            .join(ContentHashSHA256Component, Entity.id == ContentHashSHA256Component.entity_id)
            .where(ContentHashSHA256Component.hash_value == hash_value)
        )
    elif normalized_hash_type == "md5":
        stmt = (
            select(Entity)
            .join(ContentHashMD5Component, Entity.id == ContentHashMD5Component.entity_id)
            .where(ContentHashMD5Component.hash_value == hash_value)
        )
    else:
        logger.error(f"Unsupported hash type for search: {hash_type}")
        return None

    result = session.execute(stmt).scalar_one_or_none()
    return result


def add_asset_file(
    session: Session,
    filepath_on_disk: Path,
    original_filename: str,  # User-provided original filename
    mime_type: str,
    size_bytes: int,
    world_name: Optional[str] = None,  # Added world_name for file_storage
) -> Tuple[Entity, bool]:  # Returns (Entity, created_new_entity_flag)
    """
    Adds an asset file to the DAM system using content-addressable storage for a specific world.
    - Reads file content from filepath_on_disk.
    - Stores the file using file_storage.store_file, which returns a file_identifier (SHA256 hash).
    - Checks if an entity with this file_identifier (content_hash) already exists.
    - If not, creates a new Entity and associated components.
    - If yes, links the new original_filename to the existing entity if not already present.

    Args:
        session: SQLAlchemy session.
        filepath_on_disk: Path to the source file on disk.
        original_filename: Original name of the file (can be different from filepath_on_disk.name).
        mime_type: MIME type of the file.
        size_bytes: Size of the file in bytes.

    Returns:
        A tuple containing the Entity (new or existing) and a boolean indicating
        if a new Entity was created.
    """
    created_new_entity = False

    # Read file content
    try:
        file_content = filepath_on_disk.read_bytes()
    except IOError:
        logger.exception(f"Error reading file {filepath_on_disk}")
        raise

    # Dynamically import settings to ensure patched version is used in tests
    from dam.core import config as app_config  # Changed import

    name_for_lookup = world_name
    if name_for_lookup is None:
        name_for_lookup = app_config.settings.DEFAULT_WORLD_NAME

    world_config_obj = app_config.settings.get_world_config(name_for_lookup)

    # Store the file using the new service, returns the SHA256 hash (content_hash)
    # and the relative physical path suffix for CAS.
    content_hash_sha256, physical_storage_path_suffix = file_storage.store_file(  # Modified
        file_content, world_config=world_config_obj, original_filename=original_filename
    )

    # Try to find an existing entity using the SHA256 content hash
    existing_entity = find_entity_by_content_hash(session, content_hash_sha256, "sha256")

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Linking original source."
        )
        # Ensure MD5 hash is also stored for this existing entity if not already present
        # (This assumes FilePropertiesComponent and other core hashes are only added once per entity)
        md5_hash_value = calculate_md5(filepath_on_disk)  # Calculate MD5 for the source file
        existing_md5_components = get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == md5_hash_value for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
            add_component_to_entity(session, entity.id, chc_md5)
            logger.info(f"Added MD5 hash '{md5_hash_value}' to existing Entity ID {entity.id}.")

        # Check if this specific CAS FileLocationComponent already exists (it should if entity was found by this hash)
        # This is more of a sanity check or for cases where FLC might have been missed.
        existing_cas_locations = get_components(session, entity.id, FileLocationComponent)
        found_cas_location = False
        for loc in existing_cas_locations:
            if loc.storage_type == "local_cas" and loc.physical_path_or_key == physical_storage_path_suffix:
                found_cas_location = True
                logger.debug(
                    f"CAS FileLocationComponent for {physical_storage_path_suffix} already exists for Entity {entity.id}."
                )
                break
        if not found_cas_location:
            logger.warning(
                f"Entity {entity.id} found by content hash {content_hash_sha256[:12]}, "
                f"but no existing local_cas FileLocationComponent with path {physical_storage_path_suffix}. Adding one."
            )
            flc = FileLocationComponent(
                entity_id=entity.id,
                entity=entity,
                content_identifier=content_hash_sha256,
                storage_type="local_cas",
                physical_path_or_key=physical_storage_path_suffix,
                contextual_filename=original_filename,  # Store original filename with this specific CAS location
            )
            add_component_to_entity(session, entity.id, flc)

    else:  # No existing entity for this content hash
        created_new_entity = True
        entity = create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for '{original_filename}' (SHA256: {content_hash_sha256[:12]}...)."
        )

        # Core Components for new Entity
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        add_component_to_entity(session, entity.id, chc_sha256)

        md5_hash_value = calculate_md5(filepath_on_disk)
        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=md5_hash_value)
        add_component_to_entity(session, entity.id, chc_md5)
        logger.info(f"Added SHA256 and MD5 hashes for new Entity ID {entity.id}.")

        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,  # This is the primary original filename for the entity
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        add_component_to_entity(session, entity.id, fpc)

        # File Location Component for the new CAS file
        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            content_identifier=content_hash_sha256,
            storage_type="local_cas",  # Changed from "local_content_addressable"
            physical_path_or_key=physical_storage_path_suffix,
            contextual_filename=original_filename,  # original_filename for this CAS location
        )
        add_component_to_entity(session, entity.id, flc)

    # Always add OriginalSourceInfoComponent for this ingestion event
    from dam.models.original_source_info_component import OriginalSourceInfoComponent  # Import here

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id,
        entity=entity,
        original_filename=original_filename,
        original_path=str(filepath_on_disk.resolve()),
    )
    add_component_to_entity(session, entity.id, osi_comp)
    logger.info(f"Added OriginalSourceInfo for '{original_filename}' to Entity ID {entity.id}.")

    # After entity creation or retrieval, if it's an image, add perceptual hashes
    # This logic remains the same, ensuring these components are added if not already present for the entity.
    if mime_type and mime_type.startswith("image/"):
        # Get existing perceptual hashes for each type
        existing_phashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualPHashComponent)
        }
        existing_ahashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualAHashComponent)
        }
        existing_dhashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualDHashComponent)
        }

        perceptual_hashes = generate_perceptual_hashes(filepath_on_disk)

        if "phash" in perceptual_hashes and perceptual_hashes["phash"] not in existing_phashes:
            iphc = ImagePerceptualPHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"]
            )
            add_component_to_entity(session, entity.id, iphc)
            logger.info(f"Added phash '{perceptual_hashes['phash'][:12]}...' for Entity ID {entity.id}.")

        if "ahash" in perceptual_hashes and perceptual_hashes["ahash"] not in existing_ahashes:
            iahc = ImagePerceptualAHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"]
            )
            add_component_to_entity(session, entity.id, iahc)
            logger.info(f"Added ahash '{perceptual_hashes['ahash'][:12]}...' for Entity ID {entity.id}.")

        if "dhash" in perceptual_hashes and perceptual_hashes["dhash"] not in existing_dhashes:
            idhc = ImagePerceptualDHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"]
            )
            add_component_to_entity(session, entity.id, idhc)
            logger.info(f"Added dhash '{perceptual_hashes['dhash'][:12]}...' for Entity ID {entity.id}.")

    # Mark the entity as needing metadata extraction
    # The actual extraction will be handled by a system in a later stage.
    # Ensure the component is only added if it doesn't already exist (though add_component_to_entity might handle this)
    if not get_components(session, entity.id, NeedsMetadataExtractionComponent):  # Check first
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        add_component_to_entity(
            session, entity.id, marker_comp, flush=True
        )  # Flush needed to make marker visible for subsequent systems in same overall transaction if not handled by scheduler stages with new sessions
        logger.info(f"Marked Entity ID {entity.id} with NeedsMetadataExtractionComponent.")
    else:
        logger.debug(f"Entity ID {entity.id} already marked for metadata extraction or processed.")

    return entity, created_new_entity


def add_asset_reference(
    session: Session,
    filepath_on_disk: Path,  # Path to the original file, not to be copied
    original_filename: str,
    mime_type: str,
    size_bytes: int,
    world_name: Optional[str] = None,  # Added world_name for logging consistency
) -> Tuple[Entity, bool]:
    """
    Adds an asset by referencing an existing file on disk for a specific world,
    to the content-addressable storage.
    - Calculates content hashes (SHA256, MD5) from the referenced file.
    - Checks if an entity with this content (SHA256 hash) already exists.
    - If not, creates a new Entity and associated components.
    - If yes, links the new reference to the existing entity.
    - The FileLocationComponent will store the original `filepath_on_disk` and a
      special `storage_type` (e.g., "referenced_local_file").

    Args:
        session: SQLAlchemy session.
        filepath_on_disk: Absolute path to the source file on disk.
        original_filename: Original name of the file.
        mime_type: MIME type of the file.
        size_bytes: Size of the file in bytes.

    Returns:
        A tuple containing the Entity (new or existing) and a boolean indicating
        if a new Entity was created.
    """
    created_new_entity = False

    # Calculate content hashes from the existing file
    try:
        content_hash_sha256 = calculate_sha256(filepath_on_disk)
        content_hash_md5 = calculate_md5(filepath_on_disk)
    except IOError:
        logger.exception(f"Error reading file for hashing: {filepath_on_disk}")
        raise

    # Try to find an existing entity using the SHA256 hash
    existing_entity = find_entity_by_content_hash(session, content_hash_sha256, "sha256")

    if existing_entity:
        entity = existing_entity
        logger.info(
            f"Content (SHA256: {content_hash_sha256[:12]}...) for '{original_filename}' "
            f"already exists as Entity ID {entity.id}. Adding new reference."
        )
        # Ensure MD5 hash is also stored for this existing entity if not already present
        existing_md5_components = get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == content_hash_md5 for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(
                entity_id=entity.id,
                entity=entity,
                hash_value=content_hash_md5,
            )
            add_component_to_entity(session, entity.id, chc_md5)
            logger.info(f"Added MD5 hash '{content_hash_md5}' to existing Entity ID {entity.id} (found by SHA256).")

        # Check if a FileLocationComponent for this specific referenced path already exists
        # Ensure MD5 hash is also stored for this existing entity if not already present
        existing_md5_components = get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == content_hash_md5 for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=content_hash_md5)
            add_component_to_entity(session, entity.id, chc_md5)
            logger.info(f"Added MD5 hash '{content_hash_md5}' to existing Entity ID {entity.id} (found by SHA256).")

    else:  # No existing entity for this content hash
        created_new_entity = True
        entity = create_entity(session)
        logger.info(
            f"Creating new Entity ID {entity.id} for referenced file '{original_filename}' "
            f"(SHA256: {content_hash_sha256[:12]}...)."
        )

        # Core Components for new Entity
        chc_sha256 = ContentHashSHA256Component(entity_id=entity.id, entity=entity, hash_value=content_hash_sha256)
        add_component_to_entity(session, entity.id, chc_sha256)

        chc_md5 = ContentHashMD5Component(entity_id=entity.id, entity=entity, hash_value=content_hash_md5)
        add_component_to_entity(session, entity.id, chc_md5)
        logger.info(f"Added SHA256 and MD5 hashes for new Entity ID {entity.id}.")

        # File Properties Component - only add if entity is new
        # If entity existed, its FilePropertiesComponent should remain as is.
        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        add_component_to_entity(session, entity.id, fpc)

    # For both new and existing entities, manage FileLocationComponent for this specific reference
    # and always add OriginalSourceInfoComponent.

    resolved_original_path = str(filepath_on_disk.resolve())

    # Check if a FileLocationComponent for this specific referenced path already exists for this entity.
    # The unique constraint is now (entity_id, storage_type, physical_path_or_key).
    existing_location_for_this_reference = False
    existing_locations = get_components(session, entity.id, FileLocationComponent)
    for loc in existing_locations:
        if loc.storage_type == "local_reference" and loc.physical_path_or_key == resolved_original_path:
            existing_location_for_this_reference = True
            logger.info(
                f"Reference FileLocationComponent for path '{resolved_original_path}' for Entity ID {entity.id} already exists."
            )
            break

    if not existing_location_for_this_reference:
        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            content_identifier=content_hash_sha256,  # Link to the content
            storage_type="local_reference",  # Changed from "referenced_local_file"
            physical_path_or_key=resolved_original_path,  # The actual path being referenced
            contextual_filename=original_filename,  # Original filename for this reference
        )
        add_component_to_entity(session, entity.id, flc)
        logger.info(
            f"Added new FileLocationComponent (local_reference) for path '{resolved_original_path}' "
            f"to Entity ID {entity.id}."
        )

    # Always add OriginalSourceInfoComponent for this ingestion event
    from dam.models.original_source_info_component import OriginalSourceInfoComponent  # Import here

    osi_comp = OriginalSourceInfoComponent(
        entity_id=entity.id,
        entity=entity,
        original_filename=original_filename,
        original_path=resolved_original_path,
    )
    add_component_to_entity(session, entity.id, osi_comp)
    logger.info(
        f"Added OriginalSourceInfo for '{original_filename}' (path: {resolved_original_path}) to Entity ID {entity.id}."
    )

    # Add multimedia and perceptual hash components, regardless of whether new or existing entity.
    # This ensures that even if an entity was created by a non-image/video file first,
    # these components get added if a later reference *is* an image/video.
    if mime_type and mime_type.startswith("image/"):
        existing_phashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualPHashComponent)
        }
        existing_ahashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualAHashComponent)
        }
        existing_dhashes = {
            comp.hash_value for comp in get_components(session, entity.id, ImagePerceptualDHashComponent)
        }

        perceptual_hashes = generate_perceptual_hashes(filepath_on_disk)

        if "phash" in perceptual_hashes and perceptual_hashes["phash"] not in existing_phashes:
            iphc = ImagePerceptualPHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["phash"]
            )
            add_component_to_entity(session, entity.id, iphc)
        if "ahash" in perceptual_hashes and perceptual_hashes["ahash"] not in existing_ahashes:
            iahc = ImagePerceptualAHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["ahash"]
            )
            add_component_to_entity(session, entity.id, iahc)
        if "dhash" in perceptual_hashes and perceptual_hashes["dhash"] not in existing_dhashes:
            idhc = ImagePerceptualDHashComponent(
                entity_id=entity.id, entity=entity, hash_value=perceptual_hashes["dhash"]
            )
            add_component_to_entity(session, entity.id, idhc)

    # Mark the entity as needing metadata extraction
    if not get_components(session, entity.id, NeedsMetadataExtractionComponent):  # Check first
        marker_comp = NeedsMetadataExtractionComponent(entity_id=entity.id, entity=entity)
        add_component_to_entity(session, entity.id, marker_comp, flush=True)  # Flush needed
        logger.info(f"Marked Entity ID {entity.id} with NeedsMetadataExtractionComponent for referenced asset.")
    else:
        logger.debug(f"Entity ID {entity.id} already marked for metadata extraction or processed (referenced asset).")

    return entity, created_new_entity


# The _add_multimedia_components function is now removed from this file.
# Its functionality is in services.metadata_extractor.extract_and_add_multimedia_components


def find_entities_by_similar_image_hashes(
    session: Session,
    image_path: Path,
    phash_threshold: int,
    ahash_threshold: int,
    dhash_threshold: int,
) -> list[dict]:
    logger.debug(
        f"Entering find_entities_by_similar_image_hashes for image: {image_path.name}, "
        f"pTh={phash_threshold}, aTh={ahash_threshold}, dTh={dhash_threshold}"
    )
    """
    Finds entities with images similar to the provided image based on perceptual hashes.

    Args:
        session: SQLAlchemy session.
        image_path: Path to the image to compare against.
        phash_threshold: Maximum Hamming distance for pHash.
        ahash_threshold: Maximum Hamming distance for aHash.
        dhash_threshold: Maximum Hamming distance for dHash.

    Returns:
        A list of dictionaries, each containing the matched Entity,
        the type of hash that matched ('phash', 'ahash', 'dhash'),
        and the distance.
        Example: [{"entity": Entity, "match_type": "phash_match", "distance": 2, "hash_type": "phash"}, ...]
    """
    if not imagehash:
        logger.warning("ImageHash library not available. Cannot perform similarity search.")
        return []

    try:
        input_hashes = generate_perceptual_hashes(image_path)
        if not input_hashes:
            # Raise ValueError if no hashes could be generated,
            # e.g., for a non-image file or problematic image.
            msg = (
                f"Could not generate any perceptual hashes for {image_path.name}. "
                "File might not be a valid image or is unsupported."
            )
            logger.warning(msg)
            raise ValueError(msg)
    except ValueError:  # Re-raise ValueError if generate_perceptual_hashes raised it (e.g. for bad image name)
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating perceptual hashes for {image_path.name}: {e}", exc_info=True)
        raise ValueError(f"Could not process image {image_path.name} for hashing.")

    input_phash_obj = imagehash.hex_to_hash(input_hashes["phash"]) if "phash" in input_hashes else None
    input_ahash_obj = imagehash.hex_to_hash(input_hashes["ahash"]) if "ahash" in input_hashes else None
    input_dhash_obj = imagehash.hex_to_hash(input_hashes["dhash"]) if "dhash" in input_hashes else None

    source_entity_id = None
    try:
        source_content_hash = calculate_sha256(image_path)
        source_entity = find_entity_by_content_hash(session, source_content_hash, "sha256")
        if source_entity:
            source_entity_id = source_entity.id
    except Exception as e:
        logger.warning(f"Could not determine source entity for {image_path.name} to exclude from results: {e}")

    similar_entities_info = []
    # Using a set to store (entity_id, hash_type) to avoid adding duplicates if an entity matches on multiple criteria
    # but we want to report each hash type match.
    # Instead, we'll use processed_entity_ids to ensure an entity is listed once,
    # prioritizing best match.
    # The CLI currently iterates and prints all, so multiple matches for same entity are
    # fine if by different hash types.
    # For now, `processed_entity_ids` will prevent adding the same entity multiple times
    # if it matches on phash then ahash etc.
    # We will refine this if we want to show *all* hash matches for each entity.
    # The current `processed_entity_ids.add(entity.id)` means an entity is added once.
    # If we want to show an entity if it matches pHash AND aHash, we need a more complex structure.
    # The request seems to be "find similar images", so one entry per similar image is likely fine.

    # Let's collect all potential matches and then filter/sort.
    logger.debug(f"Source Entity ID for exclusion: {source_entity_id}")
    logger.debug(f"Input pHash object: {input_phash_obj}, Threshold: {phash_threshold}")
    logger.debug(f"Input aHash object: {input_ahash_obj}, Threshold: {ahash_threshold}")
    logger.debug(f"Input dHash object: {input_dhash_obj}, Threshold: {dhash_threshold}")

    potential_matches = []

    # Query for pHash
    if input_phash_obj:  # Ensure this condition is not 'False and ...'
        all_phashes_stmt = select(ImagePerceptualPHashComponent)
        db_phashes_components = session.execute(all_phashes_stmt).scalars().all()
        logger.debug(f"Found {len(db_phashes_components)} pHash components in DB.")
        for p_comp in db_phashes_components:
            logger.debug(f"Processing pHash for DB entity_id: {p_comp.entity_id}, stored pHash: {p_comp.hash_value}")
            if source_entity_id and p_comp.entity_id == source_entity_id:
                logger.debug(f"Skipping source entity {p_comp.entity_id}")
                continue
            try:
                db_phash_hex = p_comp.hash_value
                db_phash_obj = imagehash.hex_to_hash(db_phash_hex)
                distance = input_phash_obj - db_phash_obj

                logger.debug(
                    f"Entity {p_comp.entity_id}: Comparing pHash. "
                    f"Input: {input_phash_obj} (from file {image_path.name}), "
                    f"DB: {db_phash_hex} (obj: {db_phash_obj}), Distance: {distance}, "
                    f"Threshold: {phash_threshold}"
                )

                if distance <= phash_threshold:
                    entity = session.get(Entity, p_comp.entity_id)
                    if entity:
                        potential_matches.append(
                            {"entity": entity, "match_type": "phash_match", "distance": distance, "hash_type": "phash"}
                        )
                        logger.debug(f"Entity {p_comp.entity_id}: pHash MATCHED.")
                else:
                    logger.debug(f"Entity {p_comp.entity_id}: pHash MISSED (dist {distance} > th {phash_threshold}).")
            except Exception as e:
                logger.warning(f"Error comparing pHash for entity {p_comp.entity_id}: {e}")

    # Query for aHash
    if input_ahash_obj:  # Ensure this condition is not 'False and ...'
        all_ahashes_stmt = select(ImagePerceptualAHashComponent)
        db_ahashes_components = session.execute(all_ahashes_stmt).scalars().all()  # Added db_ahashes_components
        logger.debug(f"Found {len(db_ahashes_components)} aHash components in DB.")  # Added logger
        for a_comp in db_ahashes_components:  # Iterate over new variable
            if source_entity_id and a_comp.entity_id == source_entity_id:
                logger.debug(f"Skipping source entity {a_comp.entity_id} for aHash")  # Added logger
                continue
            try:
                db_ahash_obj = imagehash.hex_to_hash(a_comp.hash_value)
                distance = input_ahash_obj - db_ahash_obj
                logger.debug(  # Added logger
                    f"Entity {a_comp.entity_id}: Comparing aHash. "
                    f"Input: {input_ahash_obj}, DB: {a_comp.hash_value}, Distance: {distance}, "
                    f"Threshold: {ahash_threshold}"
                )
                if distance <= ahash_threshold:
                    entity = session.get(Entity, a_comp.entity_id)
                    if entity:
                        potential_matches.append(
                            {"entity": entity, "match_type": "ahash_match", "distance": distance, "hash_type": "ahash"}
                        )
                        logger.debug(f"Entity {a_comp.entity_id}: aHash MATCHED.")  # Added logger
                else:
                    logger.debug(
                        f"Entity {a_comp.entity_id}: aHash MISSED (dist {distance} > th {ahash_threshold})."
                    )  # Added logger
            except Exception as e:
                logger.warning(f"Error comparing aHash for entity {a_comp.entity_id}: {e}")

    # Query for dHash
    if input_dhash_obj:  # Ensure this condition is not 'False and ...'
        all_dhashes_stmt = select(ImagePerceptualDHashComponent)
        for d_comp in session.execute(all_dhashes_stmt).scalars().all():
            if source_entity_id and d_comp.entity_id == source_entity_id:
                continue  # Skip self
            try:
                db_dhash_obj = imagehash.hex_to_hash(d_comp.hash_value)
                distance = input_dhash_obj - db_dhash_obj
                if distance <= dhash_threshold:
                    entity = session.get(Entity, d_comp.entity_id)
                    if entity:
                        potential_matches.append(
                            {"entity": entity, "match_type": "dhash_match", "distance": distance, "hash_type": "dhash"}
                        )
            except Exception as e:
                logger.warning(f"Error comparing dHash for entity {d_comp.entity_id}: {e}")

    # Filter out duplicate entities, keeping the one with the best (lowest) distance
    # or by specific hash type preference if distances are equal.
    # For now, just ensure each entity appears once.
    final_matches_map = {}  # entity_id -> best_match_info
    for match in potential_matches:
        entity_id = match["entity"].id
        if entity_id not in final_matches_map or match["distance"] < final_matches_map[entity_id]["distance"]:
            final_matches_map[entity_id] = match

    similar_entities_info = list(final_matches_map.values())

    # Sort results, e.g., by entity ID or by best match (lowest distance)
    similar_entities_info.sort(key=lambda x: (x["distance"], x["entity"].id))

    return similar_entities_info
