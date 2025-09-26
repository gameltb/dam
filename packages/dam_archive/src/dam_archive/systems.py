import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, AsyncGenerator, BinaryIO, Dict, List, Optional, Tuple, Union, cast

import psutil
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    GetOrCreateEntityFromStreamCommand,
)
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.types import StreamProvider
from dam.core.world import World
from dam.functions.mime_type_functions import get_content_mime_type
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
    SystemProgressEvent,
)
from dam.utils.stream_utils import ChainedStream
from dam_fs.commands import FindEntityByFilePropertiesCommand
from dam_fs.models import FilenameComponent
from sqlalchemy import select

from . import split_detector
from .commands import (
    ClearArchiveComponentsCommand,
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    IngestArchiveCommand,
    SetArchivePasswordCommand,
    UnbindSplitArchiveCommand,
)
from .exceptions import InvalidPasswordError, PasswordRequiredError
from .main import open_archive
from .models import (
    ArchiveInfoComponent,
    ArchiveMemberComponent,
    ArchivePasswordComponent,
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)

logger = logging.getLogger(__name__)


@system(on_command=DiscoverAndBindCommand)
async def discover_and_bind_handler(
    cmd: DiscoverAndBindCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
):
    """
    Scans given paths, detects complete split archives, and creates a master entity for each.
    """
    logger.info(f"Starting discovery and binding for paths: {cmd.paths}")

    # 1. Scan file system and detect split archive parts
    all_files: List[Tuple[str, float]] = []
    for p_str in cmd.paths:
        p = Path(p_str)
        if not p.exists():
            logger.warning(f"Path does not exist, skipping: {p}")
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                for name in files:
                    full_path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(full_path)
                        all_files.append((full_path, mtime))
                    except FileNotFoundError:
                        continue
        else:
            try:
                mtime = os.path.getmtime(p)
                all_files.append((str(p), mtime))
            except FileNotFoundError:
                pass

    # Group detected parts by base_name
    parts_by_basename: Dict[str, List[Tuple[str, float, int]]] = {}
    for path, mtime in all_files:
        split_info = split_detector.detect(os.path.basename(path))
        if split_info:
            parts_by_basename.setdefault(split_info.base_name, []).append((path, mtime, split_info.part_num))

    # 2. Process each group to find complete archives
    for base_name, parts_with_meta in parts_by_basename.items():
        parts_by_num = {part_num: (path, mtime) for path, mtime, part_num in parts_with_meta}
        if not parts_by_num:
            continue

        max_part_num = max(parts_by_num.keys())
        is_complete = all(i in parts_by_num for i in range(1, max_part_num + 1))

        if is_complete:
            logger.info(f"Found complete split archive '{base_name}' with {max_part_num} parts.")

            part_entity_ids: List[int] = []
            is_valid_group = True

            # 3. Find entity IDs for each part
            sorted_parts_meta = [parts_by_num[i] for i in range(1, max_part_num + 1)]

            for path, mtime in sorted_parts_meta:
                file_uri = Path(path).as_uri()
                modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc)

                # Dispatch command to find entity
                find_cmd = FindEntityByFilePropertiesCommand(file_path=file_uri, last_modified_at=modified_at)
                try:
                    entity_id = await world.dispatch_command(find_cmd).get_one_value()
                except ValueError:
                    entity_id = None

                if entity_id:
                    part_entity_ids.append(entity_id)
                else:
                    logger.warning(
                        f"Could not find a matching entity for part '{path}' with mtime {mtime}. "
                        "Skipping assembly for this group."
                    )
                    is_valid_group = False
                    break

            # 4. Dispatch CreateMasterArchiveCommand
            if is_valid_group:
                logger.info(f"Assembling master archive for '{base_name}' with parts: {part_entity_ids}")
                master_name = f"{base_name} (Split Archive)"
                create_cmd = CreateMasterArchiveCommand(name=master_name, part_entity_ids=part_entity_ids)
                async for _ in world.dispatch_command(create_cmd):
                    pass
        else:
            logger.info(f"Split archive '{base_name}' is incomplete. Skipping assembly.")


@system(on_command=CreateMasterArchiveCommand)
async def create_master_archive_handler(
    cmd: CreateMasterArchiveCommand,
    transaction: WorldTransaction,
):
    """
    Handles the manual creation of a master entity for a split archive.
    """
    logger.info(f"Manually creating master archive '{cmd.name}' for {len(cmd.part_entity_ids)} parts.")

    # 1. Create master entity and its components
    master_entity = await transaction.create_entity()
    await transaction.add_component_to_entity(
        master_entity.id,
        FilenameComponent(filename=cmd.name, first_seen_at=datetime.now(timezone.utc)),
    )
    manifest = SplitArchiveManifestComponent()
    await transaction.add_component_to_entity(master_entity.id, manifest)

    # 2. Copy mime type from first part
    if cmd.part_entity_ids:
        first_part_id = cmd.part_entity_ids[0]
        content_mime_comp = await transaction.get_component(first_part_id, ContentMimeTypeComponent)
        if content_mime_comp:
            await transaction.add_component_to_entity(
                master_entity.id,
                ContentMimeTypeComponent(mime_type_concept_id=content_mime_comp.mime_type_concept_id),
            )

    # 3. Update all part components to link to the new master
    for part_id in cmd.part_entity_ids:
        fnc = await transaction.get_component(part_id, FilenameComponent)
        if not fnc or not fnc.filename:
            logger.warning(f"Skipping part entity {part_id} as it has no filename component.")
            continue

        split_info = split_detector.detect(fnc.filename)
        if not split_info:
            logger.warning(f"Skipping part entity {part_id} as it does not look like a split archive part.")
            continue

        part_info_comp = await transaction.get_component(part_id, SplitArchivePartInfoComponent)
        if part_info_comp:
            part_info_comp.master_entity_id = master_entity.id
        else:
            new_part_info = SplitArchivePartInfoComponent(
                part_num=split_info.part_num,
                master_entity_id=master_entity.id,
            )
            await transaction.add_component_to_entity(part_id, new_part_info)

    logger.info(f"Successfully created master entity {master_entity.id} for archive '{cmd.name}'.")


@system(on_command=UnbindSplitArchiveCommand)
async def unbind_split_archive_handler(
    cmd: UnbindSplitArchiveCommand,
    transaction: WorldTransaction,
):
    """
    Handles unbinding a split archive.
    """
    logger.info(f"Unbinding split archive master entity {cmd.master_entity_id}.")

    manifest = await transaction.get_component(cmd.master_entity_id, SplitArchiveManifestComponent)
    if not manifest:
        logger.warning(f"No split archive manifest found for entity {cmd.master_entity_id}. Nothing to unbind.")
        return

    # Delete part info from all parts
    stmt = select(SplitArchivePartInfoComponent).where(
        SplitArchivePartInfoComponent.master_entity_id == cmd.master_entity_id
    )
    result = await transaction.session.execute(stmt)
    parts_to_unbind = result.scalars().all()
    for part_info in parts_to_unbind:
        await transaction.remove_component(part_info)

    # Delete the manifest from the master
    await transaction.remove_component(manifest)

    logger.info(f"Successfully unbound archive for master entity {cmd.master_entity_id}.")


@system(on_command=SetArchivePasswordCommand)
async def set_archive_password_handler(
    cmd: SetArchivePasswordCommand,
    transaction: WorldTransaction,
) -> None:
    """
    Handles setting the password for an archive.
    """
    password_comp = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    if password_comp:
        password_comp.password = cmd.password
    else:
        password_comp = ArchivePasswordComponent(password=cmd.password)
        await transaction.add_component_to_entity(cmd.entity_id, password_comp)


@system(on_command=GetAssetStreamCommand)
async def get_archive_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> Optional[StreamProvider]:
    """
    Handles getting a stream provider for an asset that is part of an archive.
    """
    archive_member_components = await transaction.get_components(cmd.entity_id, ArchiveMemberComponent)
    if not archive_member_components:
        return None

    # This entity might be a member of multiple archives, we just need one valid one.
    for component in archive_member_components:
        target_entity_id = component.archive_entity_id
        path_in_archive = component.path_in_archive
        password_comp = await transaction.get_component(target_entity_id, ArchivePasswordComponent)
        password = password_comp.password if password_comp else None

        # Get the provider for the parent archive stream
        archive_stream_cmd = GetAssetStreamCommand(entity_id=target_entity_id)
        try:
            archive_stream_provider = await world.dispatch_command(archive_stream_cmd).get_first_non_none_value()
        except ValueError:
            archive_stream_provider = None

        if not archive_stream_provider:
            logger.warning(f"Could not get stream provider for parent archive {target_entity_id}")
            continue

        mime_type = await get_content_mime_type(transaction.session, target_entity_id)
        if not mime_type:
            logger.warning(f"Could not get mime type for parent archive {target_entity_id}")
            continue

        def provider() -> BinaryIO:
            """A closure that opens the archive and returns a stream to a member file."""
            archive_stream = archive_stream_provider()
            try:
                archive = open_archive(archive_stream, mime_type, password)
                if archive:
                    # The returned stream will be closed by the consumer.
                    # The archive object itself will be closed when the member stream is closed.
                    return archive.open_file(path_in_archive)
                else:
                    raise IOError(f"Could not open archive for entity {target_entity_id}")
            except Exception as e:
                # Ensure the parent stream is closed if opening the member fails
                archive_stream.close()
                logger.error(
                    f"Failed to open member '{path_in_archive}' from archive stream for entity {cmd.entity_id}: {e}"
                )
                raise

        return provider

    return None


@system(on_command=GetAssetFilenamesCommand)
async def get_archive_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: WorldTransaction,
) -> Optional[List[str]]:
    """
    Handles getting filenames for assets that are members of an archive.
    """
    archive_member_comps = await transaction.get_components(cmd.entity_id, ArchiveMemberComponent)
    if archive_member_comps:
        return [archive_member_comp.path_in_archive for archive_member_comp in archive_member_comps]
    return None


async def _process_archive(
    entity_id: int,
    archive_stream_provider: StreamProvider,
    cmd: IngestArchiveCommand,
    world: Annotated[World, "Resource"],
    transaction: WorldTransaction,
    is_reingestion: bool,
) -> AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent], None]:
    """
    The core extraction and event-issuing logic for an archive.
    If is_reingestion is True, it re-issues events for existing members.
    Otherwise, it performs the initial ingestion.
    """
    yield ProgressStarted()

    mime_type = await get_content_mime_type(transaction.session, entity_id)
    if not mime_type:
        yield ProgressError(message=f"Could not get mime type for archive entity {entity_id}", exception=ValueError())
        return

    stored_password_comp = await transaction.get_component(entity_id, ArchivePasswordComponent)
    passwords_to_try: List[Optional[str]] = [None]
    if stored_password_comp and stored_password_comp.password:
        passwords_to_try.insert(0, stored_password_comp.password)
    if cmd.passwords:
        passwords_to_try.extend(p for p in cmd.passwords if p)
    passwords_to_try = list(dict.fromkeys(passwords_to_try))

    archive = None
    correct_password = None

    for pwd in passwords_to_try:
        temp_archive = None
        archive_stream = archive_stream_provider()
        try:
            temp_archive = open_archive(archive_stream, mime_type, pwd)
            if temp_archive:
                correct_password = pwd
                archive = temp_archive
                logger.info(f"Successfully opened archive {entity_id} with password: {'yes' if pwd else 'no'}")
                break
        except InvalidPasswordError:
            if temp_archive:
                temp_archive.close()
            continue
        except (IOError, RuntimeError) as e:
            if temp_archive:
                temp_archive.close()
            yield ProgressError(message=f"Failed to open archive {entity_id}", exception=e)
            return
        except Exception:
            if temp_archive:
                temp_archive.close()
            archive_stream.close()
            raise

    if not archive:
        yield ProgressError(
            message=f"A password is required or the provided passwords are wrong for archive entity {entity_id}",
            exception=PasswordRequiredError(),
        )
        return

    try:
        # --- Main Logic ---
        all_members = archive.list_files()
        total_items = len(all_members)
        processed_items = 0
        total_size = 0

        member_map: Dict[str, int] = {}
        if is_reingestion:
            logger.info(f"Entity {entity_id} is being re-ingested. Mapping existing members.")
            stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
            result = await transaction.session.execute(stmt)
            members_from_db = result.scalars().all()
            member_map = {member.path_in_archive: member.entity_id for member in members_from_db}
            yield ProgressUpdate(
                total=total_items, current=processed_items, message="Re-issuing events for existing members."
            )
        else:
            logger.info(f"Entity {entity_id} is being ingested for the first time.")
            total_size = sum(m.size for m in all_members)
            yield ProgressUpdate(total=total_size, current=0, message="Starting ingestion.")

        processed_size = 0
        member_mod_times = {m.name: m.modified_at for m in all_members}

        for member_file in archive.iter_files():
            try:
                with member_file as member_stream:
                    # --- Memory-constrained stream handling ---
                    available_memory = psutil.virtual_memory().available
                    memory_limit = int(available_memory * 0.5)
                    in_memory_buffer = io.BytesIO(member_stream.read(memory_limit))
                    is_eof = not member_stream.read(1)

                    event_stream_provider: Optional[StreamProvider] = None
                    stream_for_command: BinaryIO

                    if is_eof:
                        # Read the entire content into an immutable bytes object once.
                        in_memory_buffer.seek(0)
                        buffer_content = in_memory_buffer.read()

                        # Create a new stream for the command from the bytes object.
                        stream_for_command = io.BytesIO(buffer_content)

                        # Create a provider that generates new streams from the bytes object for the event.
                        def event_provider(content: bytes = buffer_content) -> BinaryIO:
                            return io.BytesIO(content)

                        event_stream_provider = event_provider
                    else:
                        # For large files, we can't provide a re-readable event stream.
                        # The command gets a chained stream which consumes the buffer and the rest of the file stream.
                        in_memory_buffer.seek(0)
                        stream_for_command = cast(BinaryIO, ChainedStream([in_memory_buffer, member_stream]))

                    # --- Entity handling (Create vs. Re-issue) ---
                    member_entity_id: Optional[int] = None
                    if is_reingestion:
                        member_entity_id = member_map.get(member_file.name)
                        if not member_entity_id:
                            logger.warning(f"Could not find existing entity for member '{member_file.name}'. Skipping.")
                            continue
                    else:
                        # Initial ingestion: Get or create entity from stream
                        get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=stream_for_command)
                        member_entity_tuple = await world.dispatch_command(get_or_create_cmd).get_one_value()
                        if member_entity_tuple:
                            member_entity, _ = member_entity_tuple
                            member_entity_id = member_entity.id

                        if not member_entity_id:
                            raise ValueError(f"Could not get or create entity for archive member '{member_file.name}'")

                        # Add ArchiveMemberComponent for new members first to avoid race conditions.
                        member_comp = ArchiveMemberComponent(
                            archive_entity_id=entity_id,
                            path_in_archive=member_file.name,
                            modified_at=member_mod_times.get(member_file.name),
                        )
                        await transaction.add_component_to_entity(member_entity_id, member_comp)

                    # --- Event Emission ---
                    if member_entity_id:
                        yield NewEntityCreatedEvent(
                            entity_id=member_entity_id,
                            stream_provider=event_stream_provider,
                            filename=member_file.name,
                        )

                # --- Progress Update ---
                if is_reingestion:
                    processed_items += 1
                    yield ProgressUpdate(
                        total=total_items, current=processed_items, message=f"Re-issued event for '{member_file.name}'."
                    )
                else:
                    processed_size += member_file.size
                    yield ProgressUpdate(
                        total=total_size, current=processed_size, message=f"Processed '{member_file.name}'."
                    )

            except Exception as e:
                logger.error(f"Failed to process member '{member_file.name}' from archive {entity_id}: {e}")
                # Don't abort the whole process for one bad file
                continue

        # --- Finalization ---
        if not is_reingestion:
            # Save the correct password if it wasn't already stored
            if correct_password and (not stored_password_comp or correct_password != stored_password_comp.password):
                await world.dispatch_command(
                    SetArchivePasswordCommand(entity_id=entity_id, password=correct_password)
                ).get_one_value()

            info_comp = ArchiveInfoComponent(comment=archive.comment)
            await transaction.add_or_update_component(entity_id, info_comp)
            logger.info(f"Finished processing archive {entity_id}, processed {len(all_members)} members.")
            yield ProgressCompleted()
        else:
            yield ProgressCompleted(message="Finished re-issuing events for members.")

    finally:
        if archive:
            archive.close()


@system(on_command=IngestArchiveCommand)
async def ingest_archive_members_handler(
    cmd: IngestArchiveCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent], None]:
    """
    Handles processing an archive. It's the main entry point for ingestion.
    This handler determines the stream provider and then calls the main processing logic.
    """
    logger.info(f"Ingestion command received for entity {cmd.entity_id}")

    # --- Determine Stream Provider ---
    archive_stream_provider: Optional[StreamProvider] = None

    if cmd.stream_provider:
        logger.info(f"Processing archive for entity {cmd.entity_id} from provided stream provider.")
        archive_stream_provider = cmd.stream_provider
    else:
        # Case 1: Master entity for a split archive.
        manifest_comp = await transaction.get_component(cmd.entity_id, SplitArchiveManifestComponent)
        if manifest_comp:
            logger.info(f"Entity {cmd.entity_id} is a split archive master. Chaining part streams.")
            stmt = (
                select(SplitArchivePartInfoComponent)
                .where(SplitArchivePartInfoComponent.master_entity_id == cmd.entity_id)
                .order_by(SplitArchivePartInfoComponent.part_num)
            )
            result = await transaction.session.execute(stmt)
            parts = result.scalars().all()
            part_entity_ids = [part.entity_id for part in parts]
            try:
                part_stream_providers: List[StreamProvider] = []
                for part_entity_id in part_entity_ids:
                    stream_cmd = GetAssetStreamCommand(entity_id=part_entity_id)
                    provider = await world.dispatch_command(stream_cmd).get_first_non_none_value()
                    if provider:
                        part_stream_providers.append(provider)
                    else:
                        raise ValueError(f"Stream provider for part {part_entity_id} is None")

                def chained_stream_provider() -> BinaryIO:
                    streams = [p() for p in part_stream_providers]
                    return cast(BinaryIO, ChainedStream(streams))

                archive_stream_provider = chained_stream_provider
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Could not get stream for split archive part: {e}")
                yield ProgressError(message=str(e), exception=e)
                return

        # Case 2: Part of an already assembled split archive.
        part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)
        if not manifest_comp and part_info and part_info.master_entity_id:
            logger.info(
                f"Redirecting ingestion from part {cmd.entity_id} to master entity {part_info.master_entity_id}."
            )
            redirect_cmd = IngestArchiveCommand(entity_id=part_info.master_entity_id, passwords=cmd.passwords)
            async for event in world.dispatch_command(redirect_cmd):
                yield cast(SystemProgressEvent, event)
            return

        # Case 3: Part of a non-assembled split archive.
        if not manifest_comp and part_info and not part_info.master_entity_id:
            err_msg = (
                f"Entity {cmd.entity_id} is part of a non-assembled split archive. "
                "Please run 'discover-and-bind' or 'create-master' command first."
            )
            yield ProgressError(message=err_msg, exception=RuntimeError(err_msg))
            return

        # Case 4: Regular, single-file archive.
        if not manifest_comp and not part_info:
            logger.info(f"Entity {cmd.entity_id} is a single-file archive. Getting stream provider.")
            try:
                archive_stream_provider = await cmd.get_stream_provider(world)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Could not get stream for single-file archive {cmd.entity_id}: {e}")
                yield ProgressError(message=str(e), exception=e)
                return

    # --- Call Processing Logic ---
    if not archive_stream_provider:
        yield ProgressError(
            message=f"Could not determine a stream provider for entity {cmd.entity_id}", exception=ValueError()
        )
        return

    info_comp = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    is_reingestion = info_comp is not None

    async for event in _process_archive(
        cmd.entity_id, archive_stream_provider, cmd, world, transaction, is_reingestion=is_reingestion
    ):
        yield event


@system(on_command=ClearArchiveComponentsCommand)
async def clear_archive_components_handler(
    cmd: ClearArchiveComponentsCommand,
    transaction: WorldTransaction,
) -> None:
    """
    Handles clearing archive-related components from an entity and its members.
    """
    # Delete ArchiveInfoComponent from the main archive entity
    info_comp = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    if info_comp:
        await transaction.remove_component(info_comp)
        logger.info(f"Deleted ArchiveInfoComponent from entity {cmd.entity_id}")

    # Find all ArchiveMemberComponents that point to this archive entity
    stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    member_components = result.scalars().all()

    for member_comp in member_components:
        await transaction.remove_component(member_comp)
        logger.info(
            f"Deleted ArchiveMemberComponent from member entity {member_comp.entity_id} "
            f"(linked to archive {cmd.entity_id})"
        )
