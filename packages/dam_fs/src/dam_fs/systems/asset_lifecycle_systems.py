"""Defines systems for handling asset lifecycle events related to the filesystem."""

import binascii
import datetime
import logging
from typing import Annotated, Any

from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    GetOrCreateEntityFromStreamCommand,
)
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.types import FileStreamProvider
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from sqlalchemy import select

from ..commands import (
    AddFilePropertiesCommand,
    FindEntityByFilePropertiesCommand,
    FindEntityByHashCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from ..models.file_location_component import FileLocationComponent
from ..models.filename_component import FilenameComponent
from ..resources.file_storage_resource import FileStorageResource

logger = logging.getLogger(__name__)


@system(on_command=AddFilePropertiesCommand)
async def add_file_properties_handler(
    cmd: AddFilePropertiesCommand,
    transaction: WorldTransaction,
) -> None:
    """Handle the AddFilePropertiesCommand to add file properties to an entity."""
    logger.info("System handling AddFilePropertiesCommand for entity: %s", cmd.entity_id)

    # Add FilenameComponent
    fnc = FilenameComponent(filename=cmd.original_filename, first_seen_at=cmd.modified_at)
    await transaction.add_component_to_entity(cmd.entity_id, fnc)

    # Add ContentLengthComponent
    clc = ContentLengthComponent(file_size_bytes=cmd.size_bytes)
    await transaction.add_component_to_entity(cmd.entity_id, clc)


@system(on_command=FindEntityByHashCommand)
async def handle_find_entity_by_hash_command(
    cmd: FindEntityByHashCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> dict[str, Any] | None:
    """Handle the FindEntityByHashCommand to find an entity by its content hash."""
    logger.info(
        "System handling FindEntityByHashCommand for hash: %s (type: %s) in world '%s' (Req ID: %s)",
        cmd.hash_value,
        cmd.hash_type,
        world.name,
        cmd.request_id,
    )
    try:
        hash_bytes = binascii.unhexlify(cmd.hash_value)
    except binascii.Error as e:
        raise ValueError(f"Invalid hash_value format: {cmd.hash_value}") from e
    entity = await ecs_functions.find_entity_by_content_hash(transaction.session, hash_bytes, cmd.hash_type)
    if not entity:
        return None
    return {"entity_id": entity.id, "components": {}}


@system(on_command=GetAssetFilenamesCommand)
async def get_fs_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: WorldTransaction,
) -> list[str] | None:
    """Handle the GetAssetFilenamesCommand to retrieve filenames for an asset."""
    fncs = await transaction.get_components(cmd.entity_id, FilenameComponent)
    if fncs:
        return [fnc.filename for fnc in fncs if fnc.filename is not None]
    return None


@system(on_command=RegisterLocalFileCommand)
async def register_local_file_handler(
    cmd: RegisterLocalFileCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> int | None:
    """Handle the RegisterLocalFileCommand to register a local file."""
    file_path = cmd.file_path
    file_uri = file_path.as_uri()

    try:
        file_stat = file_path.stat()
        current_mtime = datetime.datetime.fromtimestamp(file_stat.st_mtime, tz=datetime.UTC)
        current_size = file_stat.st_size
    except FileNotFoundError:
        logger.warning("File not found during registration: %s", file_path)
        return None

    # Check if we've seen this file location before and if it's modified
    stmt = select(FileLocationComponent).where(FileLocationComponent.url == file_uri)
    existing_flc = (await transaction.session.execute(stmt)).scalar_one_or_none()

    if existing_flc and existing_flc.last_modified_at == current_mtime:
        logger.info("File '%s' is unchanged since last scan. Skipping.", file_path)
        return existing_flc.entity_id

    # File is new or modified, proceed with hash-based entity creation/retrieval
    stream_provider = FileStreamProvider(file_path)
    get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream_provider=stream_provider)
    result_tuple = await world.dispatch_command(get_or_create_cmd).get_one_value()
    entity, _ = result_tuple

    async with transaction.session.begin_nested():
        # Update FileLocationComponent
        if existing_flc:
            existing_flc.last_modified_at = current_mtime
        else:
            flc = FileLocationComponent(url=file_uri, last_modified_at=current_mtime)
            await transaction.add_component_to_entity(entity.id, flc)

        # Add ContentLengthComponent if it doesn't exist
        if not await transaction.get_component(entity.id, ContentLengthComponent):
            clc = ContentLengthComponent(file_size_bytes=current_size)
            await transaction.add_component_to_entity(entity.id, clc)

        # Add or update FilenameComponent
        existing_fnc = await transaction.get_component(entity.id, FilenameComponent)
        if not existing_fnc:
            fnc = FilenameComponent(filename=file_path.name, first_seen_at=current_mtime)
            await transaction.add_component_to_entity(entity.id, fnc)
        elif existing_fnc.first_seen_at and current_mtime < existing_fnc.first_seen_at:
            # We found an earlier instance of this filename, update the timestamp
            existing_fnc.first_seen_at = current_mtime

    return entity.id


@system(on_command=FindEntityByFilePropertiesCommand)
async def find_entity_by_file_properties_handler(
    cmd: FindEntityByFilePropertiesCommand,
    transaction: WorldTransaction,
) -> int | None:
    """Handle the FindEntityByFilePropertiesCommand to find an entity by file properties."""
    stmt = (
        select(FileLocationComponent.entity_id)
        .where(FileLocationComponent.url == cmd.file_path)
        .where(FileLocationComponent.last_modified_at == cmd.last_modified_at)
        .limit(1)
    )
    result = await transaction.session.execute(stmt)
    return result.scalar_one_or_none()


@system(on_command=StoreAssetsCommand)
async def store_assets_handler(
    _cmd: StoreAssetsCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handle storing assets in the file storage based on a query.

    Currently, it stores all assets that are not yet in the storage.
    """
    storage_resource = world.get_resource(FileStorageResource)
    stored_count = 0

    # A simple implementation: iterate through all entities.
    # This could be optimized with a more specific query if needed.
    all_entities_stmt = select(Entity)
    all_entities_result = await transaction.session.execute(all_entities_stmt)
    all_entities = all_entities_result.scalars().all()

    for entity in all_entities:
        sha256_comp = await transaction.get_component(entity.id, ContentHashSHA256Component)
        if not sha256_comp:
            continue  # Cannot store without a hash

        content_hash = sha256_comp.hash_value.hex()
        if storage_resource.has_file(content_hash):
            continue  # Already stored

        # Get the asset stream to store it
        asset_stream_cmd = GetAssetStreamCommand(entity_id=entity.id)
        try:
            stream_provider = await world.dispatch_command(asset_stream_cmd).get_first_non_none_value()
        except ValueError:
            stream_provider = None

        if stream_provider:
            try:
                async with stream_provider.get_stream() as stream:
                    content = stream.read()
                    storage_resource.store_file(content)
                    stored_count += 1
                    logger.info("Stored entity %s (hash: %s)", entity.id, content_hash)
            except Exception:
                logger.exception("Failed to store entity %s", entity.id)

    logger.info("Successfully stored %s new assets.", stored_count)
