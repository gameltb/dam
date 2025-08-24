"""
This service provides a high-level API for importing assets into the DAM.
It encapsulates the logic for checking for duplicates based on content hashes
and handling different import methods (e.g., copying vs. referencing).
"""

import logging
from pathlib import Path
from typing import Optional

from io import BytesIO
from dam.core.world import World
from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.transaction import EcsTransaction
from dam.models.core.entity import Entity
from dam_fs.models.file_location_component import FileLocationComponent
from dam.models.hashes.content_hash_md5_component import ContentHashMD5Component
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_fs.models.file_properties_component import FilePropertiesComponent
from dam_source.models.source_info import source_types
from dam_source.models.source_info.original_source_info_component import OriginalSourceInfoComponent
from dam.services import ecs_service, hashing_service
from dam_fs.services import file_operations

logger = logging.getLogger(__name__)


class ImportServiceError(Exception):
    """Custom exception for ImportService errors."""

    pass


async def import_stream(
    world: World,
    transaction: EcsTransaction,
    file_content: BytesIO,
    original_filename: str,
    size_bytes: int,
) -> Entity:
    """
    Imports a file from a stream into the DAM.
    """
    logger.info(f"Importing stream for: {original_filename}")

    try:
        file_content.seek(0)
        hashes = await hashing_service.calculate_hashes_from_stream_async(file_content, ["md5", "sha256"])
        sha256_hash = hashes["sha256"]
        md5_hash = hashes["md5"]
        sha256_bytes = bytes.fromhex(sha256_hash)
    except (IOError, FileNotFoundError) as e:
        raise ImportServiceError(f"Could not read or hash stream for {original_filename}: {e}") from e

    existing_entity = await transaction.find_entity_by_content_hash(sha256_bytes, "sha256")
    entity: Optional[Entity] = None

    if existing_entity:
        entity = existing_entity
        logger.info(f"Content for '{original_filename}' already exists as Entity ID {entity.id}.")
    else:
        entity = await transaction.create_entity()
        logger.info(f"Creating new Entity ID {entity.id} for '{original_filename}'.")

        chc_sha256 = ContentHashSHA256Component(hash_value=sha256_bytes)
        await transaction.add_component_to_entity(entity.id, chc_sha256)

        chc_md5 = ContentHashMD5Component(hash_value=bytes.fromhex(md5_hash))
        await transaction.add_component_to_entity(entity.id, chc_md5)

        fpc = FilePropertiesComponent(original_filename=original_filename, file_size_bytes=size_bytes)
        await transaction.add_component_to_entity(entity.id, fpc)

    if not entity:
        raise ImportServiceError("Failed to create or find entity for the asset.")

    from dam_fs.resources.file_storage_resource import FileStorageResource
    file_storage = world.get_resource(FileStorageResource)
    file_content.seek(0)
    _, physical_path = file_storage.store_file(file_content.read(), original_filename=original_filename)

    url = f"dam://local_cas/{sha256_hash}#{original_filename}"
    source_type = source_types.SOURCE_TYPE_LOCAL_FILE

    existing_flcs = await transaction.get_components(entity.id, FileLocationComponent)
    if not any(flc.url == url for flc in existing_flcs):
        flc = FileLocationComponent(content_identifier=sha256_hash, url=url, credentials=None)
        await transaction.add_component_to_entity(entity.id, flc)

    existing_osis = await transaction.get_components_by_value(
        entity.id, OriginalSourceInfoComponent, {"source_type": source_type}
    )
    if not existing_osis:
        osi = OriginalSourceInfoComponent(source_type=source_type)
        await transaction.add_component_to_entity(entity.id, osi)

    if not await transaction.get_components(entity.id, NeedsMetadataExtractionComponent):
        marker = NeedsMetadataExtractionComponent()
        await transaction.add_component_to_entity(entity.id, marker)

    return entity


async def import_local_file(
    world: World,
    transaction: EcsTransaction,
    filepath: Path,
    copy_to_storage: bool = True,
    original_filename: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> Entity:
    """
    Imports a local file into the DAM.
    """
    logger.info(f"Importing local file: {filepath} (copy: {copy_to_storage})")

    if not copy_to_storage:
        # This is a reference import, the logic is different and does not use a stream.
        # This part of the logic needs to be refactored separately.
        # For now, we will leave it as is and focus on the stream import.
        # TODO: Refactor reference import logic.
        raise NotImplementedError("Reference import not fully refactored yet.")

    try:
        if original_filename is None or size_bytes is None:
            original_filename, size_bytes = file_operations.get_file_properties(filepath)

        with open(filepath, "rb") as f:
            file_content = BytesIO(f.read())

        return await import_stream(
            world=world,
            transaction=transaction,
            file_content=file_content,
            original_filename=original_filename,
            size_bytes=size_bytes,
        )

    except (IOError, FileNotFoundError) as e:
        raise ImportServiceError(f"Could not read or hash file at {filepath}: {e}") from e
