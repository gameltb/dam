import logging  # Import logging module
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

# from dam.core.config import settings # No longer needed directly here for ASSET_STORAGE_PATH
from dam.models import Entity
from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.models.image_perceptual_hash_ahash_component import (
    ImagePerceptualAHashComponent,
)
from dam.models.image_perceptual_hash_dhash_component import (
    ImagePerceptualDHashComponent,
)
from dam.models.image_perceptual_hash_phash_component import (
    ImagePerceptualPHashComponent,
)

# Conditional import for imagehash for type hinting and direct use
try:
    import imagehash
except ImportError:
    imagehash = None


from . import file_storage  # Import the new file storage service
from .ecs_service import (
    add_component_to_entity,
    create_entity,
    get_components,
)
from .file_operations import (
    calculate_md5,  # Added for MD5 calculation
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
    # content_hash: str, # This will be derived by file_storage.store_file
    # hash_type: str = "sha256", # Assumed sha256 by file_storage.store_file
) -> Tuple[Entity, bool]:  # Returns (Entity, created_new_entity_flag)
    """
    Adds an asset file to the DAM system using content-addressable storage.
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
        # Handle file reading errors appropriately (e.g., log and raise or return error)
        logger.exception(f"Error reading file {filepath_on_disk}")  # Use logger.exception to include stack trace
        raise

    # Store the file using the new service; this also calculates the hash (file_identifier)
    # original_filename is passed to store_file for context, though not used for path generation
    file_identifier = file_storage.store_file(file_content, original_filename=original_filename)
    # The file_identifier is the SHA256 content hash

    # Try to find an existing entity using the SHA256 hash first
    existing_entity = find_entity_by_content_hash(session, file_identifier, "sha256")

    if existing_entity:
        entity = existing_entity
        # If entity found by SHA256, ensure its MD5 hash is also stored if not already
        # This handles cases where MD5 was added later or for older assets.
        md5_hash_value = calculate_md5(filepath_on_disk)
        existing_md5_components = get_components(session, entity.id, ContentHashMD5Component)
        if not any(comp.hash_value == md5_hash_value for comp in existing_md5_components):
            chc_md5 = ContentHashMD5Component(
                entity_id=entity.id,
                entity=entity,
                hash_value=md5_hash_value,
            )
            add_component_to_entity(session, entity.id, chc_md5)
            logger.info(f"Added MD5 hash '{md5_hash_value}' to existing Entity ID {entity.id} (found by SHA256).")

        # Check if this specific original_filename is already known for this entity's content
        existing_locations = get_components(session, entity.id, FileLocationComponent)
        found_location_for_original_name = False
        for loc in existing_locations:
            if loc.file_identifier == file_identifier and loc.original_filename == original_filename:
                found_location_for_original_name = True
                break

        if not found_location_for_original_name:
            # Content exists, but this original_filename is new for this content
            new_location_component = FileLocationComponent(
                entity_id=entity.id,
                entity=entity,
                file_identifier=file_identifier,
                storage_type="local_content_addressable",
                original_filename=original_filename,
            )
            add_component_to_entity(session, entity.id, new_location_component)
            logger.info(
                f"New reference for '{original_filename}' (hash: {file_identifier[:12]}...) "
                f"linked to existing Entity ID {entity.id}."
            )
        else:
            logger.info(
                f"Reference for '{original_filename}' (hash: {file_identifier[:12]}...) "
                f"already known for Entity ID {entity.id}."
            )
    else:
        created_new_entity = True
        entity = create_entity(session)  # session.flush() is done inside
        logger.info(f"New Entity ID {entity.id} for '{original_filename}' (Hash: {file_identifier[:12]}...).")

        # Content Hash Component (Primary identifier of the content - SHA256)
        chc_sha256 = ContentHashSHA256Component(
            entity_id=entity.id,
            entity=entity,
            hash_value=file_identifier,  # This is the SHA256 hash
        )
        add_component_to_entity(session, entity.id, chc_sha256)

        # MD5 Content Hash Component
        md5_hash_value = calculate_md5(filepath_on_disk)
        chc_md5 = ContentHashMD5Component(
            entity_id=entity.id,
            entity=entity,
            hash_value=md5_hash_value,
        )
        add_component_to_entity(session, entity.id, chc_md5)
        logger.info(f"Added MD5 hash '{md5_hash_value}' for new Entity ID {entity.id}.")

        # File Properties Component (Descriptive metadata)
        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,  # This is the user-provided one
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        add_component_to_entity(session, entity.id, fpc)

        # File Location Component (How to access this named instance of the content)
        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            file_identifier=file_identifier,
            storage_type="local_content_addressable",
            original_filename=original_filename,
        )
        add_component_to_entity(session, entity.id, flc)

    # After entity creation or retrieval, if it's an image, add perceptual hashes
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

    return entity, created_new_entity


def find_entities_by_similar_image_hashes(
    session: Session,
    image_path: Path,
    phash_threshold: int,
    ahash_threshold: int,
    dhash_threshold: int,
) -> list[dict]:
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
            logger.warning(f"Could not generate perceptual hashes for {image_path.name}.")
            return []
    except Exception as e:
        logger.error(f"Error generating perceptual hashes for {image_path.name}: {e}", exc_info=True)
        raise ValueError(f"Could not process image {image_path.name} for hashing.")

    input_phash_obj = imagehash.hex_to_hash(input_hashes["phash"]) if "phash" in input_hashes else None
    input_ahash_obj = imagehash.hex_to_hash(input_hashes["ahash"]) if "ahash" in input_hashes else None
    input_dhash_obj = imagehash.hex_to_hash(input_hashes["dhash"]) if "dhash" in input_hashes else None

    similar_entities_info = []
    processed_entity_ids = set()  # To avoid duplicate entities if they match on multiple hash types

    # Query for pHash
    if input_phash_obj:
        all_phashes_stmt = select(ImagePerceptualPHashComponent)
        for p_comp in session.execute(all_phashes_stmt).scalars().all():
            if p_comp.entity_id in processed_entity_ids:
                continue
            try:
                db_phash_obj = imagehash.hex_to_hash(p_comp.hash_value)
                distance = input_phash_obj - db_phash_obj
                if distance <= phash_threshold:
                    entity = session.get(Entity, p_comp.entity_id)
                    if entity:
                        similar_entities_info.append(
                            {"entity": entity, "match_type": "phash_match", "distance": distance, "hash_type": "phash"}
                        )
                        processed_entity_ids.add(entity.id)
            except Exception as e:
                logger.warning(f"Error comparing pHash for entity {p_comp.entity_id}: {e}")

    # Query for aHash
    if input_ahash_obj:
        all_ahashes_stmt = select(ImagePerceptualAHashComponent)
        for a_comp in session.execute(all_ahashes_stmt).scalars().all():
            if a_comp.entity_id in processed_entity_ids:
                continue
            try:
                db_ahash_obj = imagehash.hex_to_hash(a_comp.hash_value)
                distance = input_ahash_obj - db_ahash_obj
                if distance <= ahash_threshold:
                    entity = session.get(Entity, a_comp.entity_id)
                    if entity:
                        similar_entities_info.append(
                            {"entity": entity, "match_type": "ahash_match", "distance": distance, "hash_type": "ahash"}
                        )
                        processed_entity_ids.add(entity.id)
            except Exception as e:
                logger.warning(f"Error comparing aHash for entity {a_comp.entity_id}: {e}")

    # Query for dHash
    if input_dhash_obj:
        all_dhashes_stmt = select(ImagePerceptualDHashComponent)
        for d_comp in session.execute(all_dhashes_stmt).scalars().all():
            if d_comp.entity_id in processed_entity_ids:
                continue
            try:
                db_dhash_obj = imagehash.hex_to_hash(d_comp.hash_value)
                distance = input_dhash_obj - db_dhash_obj
                if distance <= dhash_threshold:
                    entity = session.get(Entity, d_comp.entity_id)
                    if entity:
                        similar_entities_info.append(
                            {"entity": entity, "match_type": "dhash_match", "distance": distance, "hash_type": "dhash"}
                        )
                        processed_entity_ids.add(entity.id)
            except Exception as e:
                logger.warning(f"Error comparing dHash for entity {d_comp.entity_id}: {e}")

    # Sort results, e.g., by entity ID or by best match (lowest distance)
    similar_entities_info.sort(key=lambda x: (x["distance"], x["entity"].id))

    return similar_entities_info
