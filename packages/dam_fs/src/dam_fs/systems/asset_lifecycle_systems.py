import binascii
import datetime
import io
import logging
from typing import Annotated, Any, Dict, List, Optional

from dam.commands import GetAssetFilenamesCommand
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
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
from ..utils.url_utils import get_local_path_for_url

logger = logging.getLogger(__name__)


@system(on_command=AddFilePropertiesCommand)
async def add_file_properties_handler(
    cmd: AddFilePropertiesCommand,
    transaction: EcsTransaction,
) -> None:
    """
    Handles adding file properties to an entity.
    """
    logger.info(f"System handling AddFilePropertiesCommand for entity: {cmd.entity_id}")
    fpc = FilePropertiesComponent(original_filename=cmd.original_filename, file_size_bytes=cmd.size_bytes)
    await transaction.add_component_to_entity(cmd.entity_id, fpc)


@system(on_command=FindEntityByHashCommand)
async def handle_find_entity_by_hash_command(
    cmd: FindEntityByHashCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
) -> Optional[Dict[str, Any]]:
    """
    Handles the command to find an entity by its content hash.
    """
    logger.info(
        f"System handling FindEntityByHashCommand for hash: {cmd.hash_value} (type: {cmd.hash_type}) in world '{world_config.name}' (Req ID: {cmd.request_id})"
    )
    try:
        hash_bytes = binascii.unhexlify(cmd.hash_value)
    except binascii.Error as e:
        logger.error(
            f"[QueryResult RequestID: {cmd.request_id}] Invalid hex string for hash_value '{cmd.hash_value}': {e}"
        )
        raise ValueError(f"Invalid hash_value format: {cmd.hash_value}") from e
    entity = await ecs_functions.find_entity_by_content_hash(transaction.session, hash_bytes, cmd.hash_type)
    if not entity:
        return None

    components: Dict[str, List[Dict[str, Any]]] = {}
    entity_details_dict: Optional[Dict[str, Any]] = {
        "entity_id": entity.id,
        "components": components,
    }
    return entity_details_dict


@system(on_command=GetAssetFilenamesCommand)
async def get_fs_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: EcsTransaction,
) -> Optional[List[str]]:
    """
    Handles getting filenames for assets with FilePropertiesComponent.
    """
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
    """
    Handles registering a local file, creating an entity and components.
    If entity exists (by hash), it adds new source info.
    If entity is new, it adds all components.
    """
    file_path = cmd.file_path
    with open(file_path, "rb") as f:
        file_content_stream = io.BytesIO(f.read())

    get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=file_content_stream)
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
    """
    Finds an entity by its local file path and modification time.
    """
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
    entity_id = result.scalar_one_or_none()
    return entity_id


@system(on_command=StoreAssetsCommand)
async def store_assets_handler(
    cmd: StoreAssetsCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
) -> None:
    """
    Handles storing assets in the content-addressable storage.
    """
    if cmd.query == "local_not_stored":
        stmt = select(FileLocationComponent).where(FileLocationComponent.url.startswith("file:///"))
        result = await transaction.session.execute(stmt)
        local_files = result.scalars().all()

        stored_count = 0
        for local_file_loc in local_files:
            # Check if a CAS location already exists
            all_locations = await transaction.get_components(local_file_loc.entity_id, FileLocationComponent)
            has_cas_location = any(loc.url.startswith("cas://") for loc in all_locations)

            if has_cas_location:
                continue

            # Store the file
            try:
                local_path = get_local_path_for_url(local_file_loc.url)
                if not local_path or not local_path.exists():
                    logger.warning(f"Local file for {local_file_loc.url} not found, skipping store.")
                    continue

                with open(local_path, "rb") as f:
                    content = f.read()

                storage_resource = world.get_resource(FileStorageResource)
                content_hash, _ = storage_resource.store_file(content)

                cas_url = f"cas://{content_hash}"
                cas_flc = FileLocationComponent(url=cas_url)
                await transaction.add_component_to_entity(local_file_loc.entity_id, cas_flc)
                stored_count += 1
                logger.info(f"Stored entity {local_file_loc.entity_id} to {cas_url}")

            except Exception as e:
                logger.error(f"Failed to store entity {local_file_loc.entity_id}: {e}", exc_info=True)

        logger.info(f"Successfully stored {stored_count} new assets.")

    else:
        logger.warning(f"Query '{cmd.query}' not supported by store_assets_handler.")
