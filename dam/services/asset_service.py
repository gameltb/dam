from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.core.config import settings  # For ASSET_STORAGE_PATH
from dam.models import Entity
from dam.models.content_hash_component import ContentHashComponent
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.models.image_perceptual_hash_component import (
    ImagePerceptualHashComponent,
)  # Added

from .ecs_service import (
    add_component_to_entity,
    create_entity,
    get_components,
)  # Added ECS functions
from .file_operations import (
    generate_perceptual_hashes,
    store_file_locally,
)  # Added generate_perceptual_hashes


def find_entity_by_content_hash(session: Session, hash_value: str, hash_type: str = "sha256") -> Optional[Entity]:
    """
    Finds an entity by its content hash.

    Args:
        session: SQLAlchemy session.
        hash_value: The hash value to search for.
        hash_type: The type of hash (e.g., "sha256").

    Returns:
        The Entity if found, otherwise None.
    """
    stmt = (
        select(Entity)
        .join(ContentHashComponent, Entity.id == ContentHashComponent.entity_id)
        .where(ContentHashComponent.hash_type == hash_type)
        .where(ContentHashComponent.hash_value == hash_value)
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result


def add_asset_file(
    session: Session,
    filepath_on_disk: Path,  # The actual path of the source file
    original_filename: str,
    mime_type: str,
    size_bytes: int,
    content_hash: str,
    hash_type: str = "sha256",  # Assuming SHA256 for content hash
) -> Tuple[Entity, bool]:  # Returns (Entity, created_new_entity_flag)
    """
    Adds an asset file to the DAM system.
    - Checks if an entity with the given content_hash already exists.
    - If not, creates a new Entity and associated components.
    - If yes, links the new file location to the existing entity if path is different.
    - Simulates file storage.

    Args:
        session: SQLAlchemy session.
        filepath_on_disk: Path to the source file on disk.
        original_filename: Original name of the file.
        mime_type: MIME type of the file.
        size_bytes: Size of the file in bytes.
        content_hash: SHA256 hash of the file content.
        hash_type: Type of content hash (default "sha256").

    Returns:
        A tuple containing the Entity (new or existing) and a boolean indicating
        if a new Entity was created.
    """
    created_new_entity = False
    existing_entity = find_entity_by_content_hash(session, content_hash, hash_type)

    if existing_entity:
        entity = existing_entity
        entity = existing_entity
        # Check if this specific filepath is already known for this entity
        # Use get_components from ecs_service
        existing_locations = get_components(session, entity.id, FileLocationComponent)
        found_location = False
        for loc in existing_locations:
            if loc.filepath == filepath_on_disk.as_posix():
                found_location = True
                break

        if not found_location:
            _ = store_file_locally(filepath_on_disk, Path(settings.ASSET_STORAGE_PATH), content_hash)
            new_location_component = FileLocationComponent(
                # entity_id and entity are required by __init__ due to BaseComponent kw_only
                entity_id=entity.id,
                entity=entity,
                filepath=filepath_on_disk.as_posix(),
                storage_type="local_source_link",  # Indicate it's a new link to existing content
            )
            add_component_to_entity(session, entity.id, new_location_component)
            print(f"New file location '{filepath_on_disk}' linked to existing Entity ID {entity.id}.")
        else:
            print(f"File location '{filepath_on_disk}' already known for Entity ID {entity.id}.")
    else:
        created_new_entity = True
        entity = create_entity(session)  # Use ECS service; session.flush() is done inside
        print(f"New Entity ID {entity.id} created for file '{original_filename}'.")

        # Content Hash Component
        chc = ContentHashComponent(
            entity_id=entity.id,
            entity=entity,
            hash_type=hash_type,
            hash_value=content_hash,
        )
        add_component_to_entity(session, entity.id, chc)

        # File Properties Component
        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            entity=entity,
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type,
        )
        add_component_to_entity(session, entity.id, fpc)

        # File Location Component
        _ = store_file_locally(filepath_on_disk, Path(settings.ASSET_STORAGE_PATH), content_hash)
        flc = FileLocationComponent(
            entity_id=entity.id,
            entity=entity,
            filepath=filepath_on_disk.as_posix(),
            storage_type="local_source_initial",
        )
        add_component_to_entity(session, entity.id, flc)

    # After entity creation or retrieval, if it's an image, add perceptual hashes
    if mime_type and mime_type.startswith("image/"):
        # Get existing perceptual hash types for this entity
        existing_phash_components = get_components(session, entity.id, ImagePerceptualHashComponent)
        existing_phash_types = {comp.hash_type for comp in existing_phash_components}

        perceptual_hashes = generate_perceptual_hashes(filepath_on_disk)
        for phash_type, phash_value in perceptual_hashes.items():
            if phash_type not in existing_phash_types:
                iphc = ImagePerceptualHashComponent(
                    entity_id=entity.id,
                    entity=entity,
                    hash_type=phash_type,
                    hash_value=phash_value,
                )
                add_component_to_entity(session, entity.id, iphc)
                print(f"Added {phash_type} '{phash_value}' for Entity ID {entity.id}.")

    return entity, created_new_entity
