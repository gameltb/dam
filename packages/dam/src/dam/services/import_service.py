"""
This service provides a high-level API for importing assets into the DAM.
It encapsulates the logic for checking for duplicates based on content hashes
and handling different import methods (e.g., copying vs. referencing).
"""

import logging
from pathlib import Path
from typing import Optional

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.world import World
from dam.models.core.entity import Entity
from dam.models.core.file_location_component import FileLocationComponent
from dam.models.hashes.content_hash_md5_component import ContentHashMD5Component
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.hashes.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from dam.models.hashes.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from dam.models.hashes.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.models.source_info import source_types
from dam.models.source_info.original_source_info_component import OriginalSourceInfoComponent
from dam.services import ecs_service, file_operations, hashing_service

logger = logging.getLogger(__name__)


class ImportServiceError(Exception):
    """Custom exception for ImportService errors."""

    pass


async def import_local_file(
    world: World,
    filepath: Path,
    copy_to_storage: bool = True,
    original_filename: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> Entity:
    """
    Imports a local file into the DAM.
    """
    logger.info(f"Importing local file: {filepath} (copy: {copy_to_storage})")

    try:
        # Use provided metadata if available, otherwise calculate it.
        if original_filename is None or size_bytes is None:
            original_filename, size_bytes = file_operations.get_file_properties(filepath)

        with open(filepath, "rb") as f:
            hashes = await hashing_service.calculate_hashes_from_stream_async(f, ["md5", "sha256"])

        sha256_hash = hashes["sha256"]
        md5_hash = hashes["md5"]
        sha256_bytes = bytes.fromhex(sha256_hash)

    except (IOError, FileNotFoundError) as e:
        raise ImportServiceError(f"Could not read or hash file at {filepath}: {e}") from e

    async with world.db_session_maker() as session:
        existing_entity = await ecs_service.find_entity_by_content_hash(session, sha256_bytes, "sha256")
        entity: Optional[Entity] = None
        created_new_entity = False

        if existing_entity:
            entity = existing_entity
            logger.info(f"Content for '{original_filename}' already exists as Entity ID {entity.id}.")
        else:
            created_new_entity = True
            entity = await ecs_service.create_entity(session)
            logger.info(f"Creating new Entity ID {entity.id} for '{original_filename}'.")

            # Add core components for a new asset
            chc_sha256 = ContentHashSHA256Component(hash_value=sha256_bytes)
            await ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

            chc_md5 = ContentHashMD5Component(hash_value=bytes.fromhex(md5_hash))
            await ecs_service.add_component_to_entity(session, entity.id, chc_md5)

            fpc = FilePropertiesComponent(original_filename=original_filename, file_size_bytes=size_bytes)
            await ecs_service.add_component_to_entity(session, entity.id, fpc)

        if not entity:
            raise ImportServiceError("Failed to create or find entity for the asset.")

        # Handle File Location and Original Source
        if copy_to_storage:
            from dam.resources.file_storage_resource import FileStorageResource

            file_storage = world.get_resource(FileStorageResource)
            try:
                file_content = await file_operations.read_file_async(filepath)
            except IOError as e:
                raise ImportServiceError(f"Failed to read file for storage: {e}") from e

            _, physical_path = file_storage.store_file(file_content, original_filename=original_filename)

            url = f"dam://local_cas/{sha256_hash}#{original_filename}"
            source_type = source_types.SOURCE_TYPE_LOCAL_FILE
        else:  # By reference
            resolved_path = str(filepath.resolve())
            url = f"dam://local_reference/{resolved_path}#{original_filename}"
            source_type = source_types.SOURCE_TYPE_REFERENCED_FILE

        # Add FileLocationComponent, checking for duplicates
        existing_flcs = await ecs_service.get_components(session, entity.id, FileLocationComponent)
        if not any(flc.url == url for flc in existing_flcs):
            flc = FileLocationComponent(content_identifier=sha256_hash, url=url, credentials=None)
            await ecs_service.add_component_to_entity(session, entity.id, flc)

        # Add OriginalSourceInfoComponent, checking for duplicates
        existing_osis = await ecs_service.get_components_by_value(
            session, entity.id, OriginalSourceInfoComponent, {"source_type": source_type}
        )
        if not existing_osis:
            osi = OriginalSourceInfoComponent(source_type=source_type)
            await ecs_service.add_component_to_entity(session, entity.id, osi)

        # Add perceptual hashes for images
        mime_type = await file_operations.get_mime_type_async(filepath)
        if mime_type.startswith("image/"):
            hashes = await hashing_service.generate_perceptual_hashes_async(filepath)
            if hashes.get("phash"):
                comp = ImagePerceptualPHashComponent(hash_value=bytes.fromhex(hashes["phash"]))
                await ecs_service.add_component_to_entity(session, entity.id, comp)
            if hashes.get("ahash"):
                comp = ImagePerceptualAHashComponent(hash_value=bytes.fromhex(hashes["ahash"]))
                await ecs_service.add_component_to_entity(session, entity.id, comp)
            if hashes.get("dhash"):
                comp = ImagePerceptualDHashComponent(hash_value=bytes.fromhex(hashes["dhash"]))
                await ecs_service.add_component_to_entity(session, entity.id, comp)

        # Mark for metadata extraction
        if not await ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
            marker = NeedsMetadataExtractionComponent()
            await ecs_service.add_component_to_entity(session, entity.id, marker)

        await session.commit()
        return entity
