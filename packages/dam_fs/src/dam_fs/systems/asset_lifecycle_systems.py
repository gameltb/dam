import binascii
import datetime
import logging
from typing import Annotated, Any, Dict, List, Optional

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_source.models.source_info import source_types
from dam_source.models.source_info.original_source_info_component import (
    OriginalSourceInfoComponent,
)
from sqlalchemy import select

from ..commands import (
    AddFilePropertiesCommand,
    FindEntityByFilePropertiesCommand,
    FindEntityByHashCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from ..models.file_location_component import FileLocationComponent
from ..models.file_properties_component import FilePropertiesComponent
from ..resources.file_storage_resource import FileStorageResource

logger = logging.getLogger(__name__)


@system(on_command=AddFilePropertiesCommand)
async def add_file_properties_handler(
    cmd: AddFilePropertiesCommand,
    transaction: EcsTransaction,
) -> None:
    logger.info(f"System handling AddFilePropertiesCommand for entity: {cmd.entity_id}")
    fpc = FilePropertiesComponent(original_filename=cmd.original_filename, file_size_bytes=cmd.size_bytes)
    await transaction.add_component_to_entity(cmd.entity_id, fpc)


@system(on_command=FindEntityByHashCommand)
async def handle_find_entity_by_hash_command(
    cmd: FindEntityByHashCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
) -> Optional[Dict[str, Any]]:
    logger.info(
        f"System handling FindEntityByHashCommand for hash: {cmd.hash_value} (type: {cmd.hash_type}) in world '{world_config.name}' (Req ID: {cmd.request_id})"
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
    transaction: EcsTransaction,
) -> Optional[List[str]]:
    file_props = await transaction.get_component(cmd.entity_id, FilePropertiesComponent)
    if file_props and file_props.original_filename:
        return [file_props.original_filename]
    return None


@system(on_command=RegisterLocalFileCommand)
async def register_local_file_handler(
    cmd: RegisterLocalFileCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> int:
    file_path = cmd.file_path
    with open(file_path, "rb") as f:
        get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=f)
        command_result = await world.dispatch_command(get_or_create_cmd)

    entity, _ = command_result.get_one_value()

    async with transaction.session.begin_nested():
        existing_fpc = await transaction.get_component(entity.id, FilePropertiesComponent)
        if not existing_fpc:
            mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.timezone.utc)
            fpc = FilePropertiesComponent(
                original_filename=file_path.name,
                file_size_bytes=file_path.stat().st_size,
                file_modified_at=mod_time,
            )
            await transaction.add_component_to_entity(entity.id, fpc)

        file_uri = file_path.as_uri()
        existing_flcs = await transaction.get_components_by_value(entity.id, FileLocationComponent, {"url": file_uri})
        if not existing_flcs:
            flc = FileLocationComponent(url=file_uri)
            await transaction.add_component_to_entity(entity.id, flc)
            osi = OriginalSourceInfoComponent(source_type=source_types.SOURCE_TYPE_LOCAL_FILE)
            await transaction.add_component_to_entity(entity.id, osi)
    return entity.id


@system(on_command=FindEntityByFilePropertiesCommand)
async def find_entity_by_file_properties_handler(
    cmd: FindEntityByFilePropertiesCommand,
    transaction: EcsTransaction,
) -> Optional[int]:
    stmt = (
        select(FileLocationComponent.entity_id)
        .join(
            FilePropertiesComponent,
            FileLocationComponent.entity_id == FilePropertiesComponent.entity_id,
        )
        .where(FileLocationComponent.url == cmd.file_path)
        .where(FilePropertiesComponent.file_modified_at == cmd.file_modified_at)
        .limit(1)
    )
    result = await transaction.session.execute(stmt)
    return result.scalar_one_or_none()


@system(on_command=StoreAssetsCommand)
async def store_assets_handler(
    cmd: StoreAssetsCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handles storing assets in the file storage based on a query.
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
        asset_stream_result = await world.dispatch_command(asset_stream_cmd)

        try:
            asset_stream = asset_stream_result.get_first_non_none_value()
        except ValueError:
            asset_stream = None

        if asset_stream:
            try:
                content = asset_stream.read()
                storage_resource.store_file(content)
                stored_count += 1
                logger.info(f"Stored entity {entity.id} (hash: {content_hash})")
            except Exception as e:
                logger.error(f"Failed to store entity {entity.id}: {e}", exc_info=True)
            finally:
                asset_stream.close()

    logger.info(f"Successfully stored {stored_count} new assets.")
