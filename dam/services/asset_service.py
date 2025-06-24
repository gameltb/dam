from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from dam.models import Entity
from dam.models.content_hash_component import ContentHashComponent
from dam.models.file_location_component import FileLocationComponent
from dam.models.file_properties_component import FilePropertiesComponent
from dam.core.config import settings # For ASSET_STORAGE_PATH

from .file_operations import store_file_locally # For simulated storage

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
    filepath_on_disk: Path, # The actual path of the source file
    original_filename: str,
    mime_type: str,
    size_bytes: int,
    content_hash: str,
    hash_type: str = "sha256" # Assuming SHA256 for content hash
) -> Tuple[Entity, bool]: # Returns (Entity, created_new_entity_flag)
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
        # Check if this specific filepath is already known for this entity
        stmt_loc = (
            select(FileLocationComponent)
            .where(FileLocationComponent.entity_id == entity.id)
            .where(FileLocationComponent.filepath == filepath_on_disk.as_posix()) # Store as string
        )
        existing_location = session.execute(stmt_loc).scalar_one_or_none()
        if not existing_location:
            # Simulate storage and get relative path for component
            # For simulation, storage_base_path is from settings.
            # The actual 'filepath' stored in component should be relative to some root
            # or an identifier that can be resolved later.
            # For local simulation, store_file_locally can return a path relative to settings.ASSET_STORAGE_PATH

            # This simulation needs a base path. Let's assume settings.ASSET_STORAGE_PATH
            # is the root for our simulated storage.
            # The component should store a path that makes sense within the DAM's context,
            # not necessarily the absolute source path.

            # For now, let's assume the 'filepath' in FileLocationComponent is the *original* source path
            # until proper storage simulation that returns a "managed path" is implemented.
            # This is a simplification for the current step.
            # A better approach: store_file_locally would return a "managed relative path".
            _ = store_file_locally(filepath_on_disk, Path(settings.ASSET_STORAGE_PATH), content_hash)
            # The path stored in FileLocationComponent should be the one within our managed storage.
            # For now, using original_filename as a placeholder for the stored path, or a hash-based one.
            # Let's use the original path as "where it was found" for now.
            new_location = FileLocationComponent(
                entity_id=entity.id, # type: ignore
                entity=entity,       # type: ignore
                filepath=filepath_on_disk.as_posix(), # Storing the original path for now
                storage_type="local_source" # Indicating it's the source path
            )
            session.add(new_location)
            print(f"New file location '{filepath_on_disk}' linked to existing Entity ID {entity.id}.")
        else:
            print(f"File location '{filepath_on_disk}' already known for Entity ID {entity.id}.")
    else:
        created_new_entity = True
        # Create new Entity and all components
        entity = Entity()
        session.add(entity)
        # Must flush to get entity.id for components if not passing entity object directly
        session.flush()

        # Content Hash Component
        chc = ContentHashComponent(
            entity_id=entity.id, # type: ignore
            entity=entity,       # type: ignore
            hash_type=hash_type,
            hash_value=content_hash
        )
        session.add(chc)

        # File Properties Component
        fpc = FilePropertiesComponent(
            entity_id=entity.id, # type: ignore
            entity=entity,       # type: ignore
            original_filename=original_filename,
            file_size_bytes=size_bytes,
            mime_type=mime_type
        )
        session.add(fpc)

        # File Location Component (after simulated storage)
        # See notes above about what path to store. Using original path for now.
        _ = store_file_locally(filepath_on_disk, Path(settings.ASSET_STORAGE_PATH), content_hash)
        flc = FileLocationComponent(
            entity_id=entity.id, # type: ignore
            entity=entity,       # type: ignore
            filepath=filepath_on_disk.as_posix(), # Storing the original path
            storage_type="local_source"
        )
        session.add(flc)
        print(f"New Entity ID {entity.id} created for file '{original_filename}'.")

    return entity, created_new_entity
