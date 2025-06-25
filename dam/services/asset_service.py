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

        # Check if a FileLocationComponent for this content (file_identifier) already exists for this entity.
        # The UniqueConstraint is on (entity_id, file_identifier).
        # If it exists, we don't add another FileLocationComponent for this content under this entity,
        # even if original_filename is different. The entity is already linked to this content.
        # We could potentially update the FilePropertiesComponent if the new original_filename is preferred,
        # or store multiple original filenames in FilePropertiesComponent or a new component,
        # but for now, we avoid duplicate FileLocationComponent for the same (entity, content_hash).

        existing_location_for_content = False
        existing_locations = get_components(session, entity.id, FileLocationComponent)
        for loc in existing_locations:
            if loc.file_identifier == file_identifier:
                existing_location_for_content = True
                # Optionally, log if the original_filename for this existing location is different
                if loc.original_filename != original_filename:
                    logger.info(
                        f"Content (hash: {file_identifier[:12]}...) for Entity ID {entity.id} "
                        f"already has a FileLocationComponent (original_filename: '{loc.original_filename}'). "
                        f"New reference attempt with original_filename '{original_filename}' "
                        "will not create a new FileLocationComponent."
                    )
                else:
                    logger.info(
                        f"FileLocationComponent for content (hash: {file_identifier[:12]}...) and "
                        f"original_filename '{original_filename}' already exists for Entity ID {entity.id}."
                    )
                break

        if not existing_location_for_content:
            # This case should ideally not be hit if find_entity_by_content_hash found an entity,
            # as that entity must have had a FileLocationComponent that led to its ContentHashSHA256Component.
            # However, to be safe, or if logic changes, we add one.
            # More likely, this block is for when the entity exists but somehow this specific
            # file_identifier (content) wasn't linked via FileLocationComponent yet, which is unusual.
            # The primary scenario for an existing entity is that it *does* have a FileLocationComponent
            # for this file_identifier.
            # This logic branch now primarily handles adding a FileLocationComponent if, hypothetically,
            # an entity was created for a content hash but didn't get a FileLocationComponent.
            # The more common case when `existing_entity` is true, is that
            # `existing_location_for_content` will also be true.

            # The original logic was: if a FileLocationComponent with this original_filename doesn't exist, add one.
            # This conflicted with the UniqueConstraint on (entity_id, file_identifier).
            # The new logic: if no FileLocationComponent for this file_identifier exists for this entity, add one.
            # This should be rare if the entity was found via its content hash linked through a FileLocation.

            # Let's stick to the original intent for differing original_filenames IF the constraint were different.
            # Given the constraint `UniqueConstraint("entity_id", "file_identifier")`,
            # we simply ensure one such component exists. If `find_entity_by_content_hash`
            # returned an entity, it implies such a link (via ContentHashSHA256 -> Entity,
            # and ContentHashSHA256 is typically added alongside FileLocationComponent)
            # must exist.

            # The previous code `if not found_location_for_original_name:` would attempt to add
            # a new FileLocationComponent if the original_filename was different, leading to the IntegrityError.
            # Now, if `existing_entity` is true, we assume the link is established.
            # The critical part is that `file_storage.store_file` ensures content is stored once.
            # `add_asset_file` ensures an Entity for that content exists once.
            # The FileLocationComponent links this Entity to its stored content using file_identifier.
            # The UniqueConstraint ensures this link (Entity <-> Content) is singular.
            # The `original_filename` on FileLocationComponent is descriptive for that singular link.
            # If multiple original filenames point to the same content, they should resolve to the same Entity,
            # and that Entity will have one FileLocationComponent pointing to the stored content.
            # The FilePropertiesComponent, however, *can* be different per original_filename if we choose,
            # but currently, it's also one per entity.

            # If the goal is to track every original_filename that pointed to the same content,
            # FileLocationComponent would need its unique constraint changed to include original_filename,
            # OR another component would track original filenames.
            # With the current constraint, we assume the entity is known by its content, and one FileLocationComponent
            # suffices. We log that we are not adding a duplicate FileLocation.
            pass  # No action needed if existing_location_for_content is true.
            # If it were false, it means the entity was found by hash, but no FLC for that hash.
            # This is an inconsistency.
            # For now, we assume if existing_entity, then existing_location_for_content is also true.
            # The test failure indicates this assumption might be violated or the setup logic is tricky.

            # Let's refine: if existing_entity is found, it *must* have a ContentHashSHA256Component.
            # This component is added alongside a FileLocationComponent when an entity is first created.
            # So, a FileLocationComponent with this file_identifier for this entity should exist.
            # The only thing to check is if the user is trying to "add" the same content
            # again but with a *different original_filename*.
            # The current FileLocationComponent model can only store one original_filename for that link.
            # We could choose to update it, or ignore the new original_filename.
            # The previous code `if not found_location_for_original_name:` would add a new FLC, causing error.
            #
            # If we want to allow different original_filenames to be associated with the same content for an entity,
            # the UniqueConstraint on FileLocationComponent must change to include `original_filename`.
            # `UniqueConstraint("entity_id", "file_identifier", "original_filename", ...)`
            #
            # Assuming the current constraint `UniqueConstraint("entity_id", "file_identifier")` is intended:
            # If an entity for this content already exists, we don't need to do anything with FileLocationComponent,
            # as one should already exist. The `original_filename` on that existing component is what it is.
            # We could log that an attempt was made to add the same content with a different original_filename.
            if not existing_location_for_content:
                # This path implies an inconsistency: entity found by content hash,
                # but no FLC for that content.
                # This might happen if an entity was created, ContentHash component added,
                # but FLC creation failed/skipped.
                # Or, if find_entity_by_content_hash joins through a different path not involving FLC directly.
                # For robustness, if no FLC exists for this content_id and entity_id, create it.
                logger.warning(
                    f"Entity ID {entity.id} found for content {file_identifier[:12]}, "
                    f"but no existing FileLocationComponent for this specific file_identifier. Adding one."
                )
                new_location_component = FileLocationComponent(
                    entity_id=entity.id,
                    entity=entity,
                    file_identifier=file_identifier,
                    storage_type="local_content_addressable",
                    original_filename=original_filename,  # Use the current original_filename being processed
                )
                add_component_to_entity(session, entity.id, new_location_component)
                logger.info(
                    f"Added FileLocationComponent for '{original_filename}' (hash: {file_identifier[:12]}...) "
                    f"to existing Entity ID {entity.id}."
                )
            # If existing_location_for_content is true, we've already logged it. No new FLC is added.

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
    if input_phash_obj:
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

    # Query for aHash (temporarily commented out for pHash debugging)
    if False and input_ahash_obj:  # Temporarily disable
        all_ahashes_stmt = select(ImagePerceptualAHashComponent)
        for a_comp in session.execute(all_ahashes_stmt).scalars().all():
            if source_entity_id and a_comp.entity_id == source_entity_id:
                continue  # Skip self
            try:
                db_ahash_obj = imagehash.hex_to_hash(a_comp.hash_value)
                distance = input_ahash_obj - db_ahash_obj
                if distance <= ahash_threshold:
                    entity = session.get(Entity, a_comp.entity_id)
                    if entity:
                        potential_matches.append(
                            {"entity": entity, "match_type": "ahash_match", "distance": distance, "hash_type": "ahash"}
                        )
            except Exception as e:
                logger.warning(f"Error comparing aHash for entity {a_comp.entity_id}: {e}")

    # Query for dHash
    if input_dhash_obj:
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
