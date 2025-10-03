import io
import logging
from contextlib import asynccontextmanager
from typing import (
    Annotated,
    Any,
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterator,
    BinaryIO,
    List,
    Optional,
    Union,
    cast,
)

import psutil
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    GetOrCreateEntityFromStreamCommand,
)
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.types import CallableStreamProvider, StreamProvider
from dam.core.world import World
from dam.functions.mime_type_functions import get_content_mime_type
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
    SystemProgressEvent,
)
from dam.system_events.requests import InformationRequest, PasswordRequest
from dam.utils.stream_utils import ChainedStream
from sqlalchemy import select

from ..commands.ingestion import (
    CheckArchiveCommand,
    ClearArchiveComponentsCommand,
    IngestArchiveCommand,
    ReissueArchiveMemberEventsCommand,
)
from ..commands.password import SetArchivePasswordCommand
from ..exceptions import InvalidPasswordError, PasswordRequiredError
from ..main import open_archive
from ..models import (
    ArchiveInfoComponent,
    ArchiveMemberComponent,
    ArchivePasswordComponent,
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)

logger = logging.getLogger(__name__)


@system(on_command=CheckArchiveCommand)
async def check_archive_handler(
    cmd: CheckArchiveCommand,
    transaction: WorldTransaction,
) -> bool:
    """Checks if the ArchiveInfoComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    return component is not None


@system(on_command=ReissueArchiveMemberEventsCommand)
async def reissue_archive_member_events_handler(
    cmd: ReissueArchiveMemberEventsCommand,
    transaction: WorldTransaction,
) -> AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent], None]:
    """
    Handles re-issuing NewEntityCreatedEvent events for all members of an existing archive.
    """
    yield ProgressStarted()

    stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    members_from_db = result.scalars().all()

    total_items = len(members_from_db)
    yield ProgressUpdate(total=total_items, current=0, message="Re-issuing events for existing members.")

    for i, member in enumerate(members_from_db):
        # We don't need to provide a stream provider here.
        # Downstream systems that need the stream can use GetAssetStreamCommand,
        # and the get_archive_asset_stream_handler will provide it.
        yield NewEntityCreatedEvent(
            entity_id=member.entity_id,
            filename=member.path_in_archive,
        )
        yield ProgressUpdate(
            total=total_items, current=i + 1, message=f"Re-issued event for '{member.path_in_archive}'."
        )

    yield ProgressCompleted(message="Finished re-issuing events for members.")


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
        all_providers = await world.dispatch_command(archive_stream_cmd).get_all_results()
        valid_providers = [p for p in all_providers if p is not None]
        if not valid_providers:
            logger.warning(f"Could not get stream provider for parent archive {target_entity_id}")
            continue
        archive_stream_provider = valid_providers[0]

        if not archive_stream_provider:
            logger.warning(f"Could not get stream provider for parent archive {target_entity_id}")
            continue

        mime_type = await get_content_mime_type(transaction.session, target_entity_id)
        if not mime_type:
            logger.warning(f"Could not get mime type for parent archive {target_entity_id}")
            continue

        class ArchiveMemberStreamProvider(StreamProvider):
            @asynccontextmanager
            async def get_stream(self) -> AsyncIterator[BinaryIO]:
                archive = await open_archive(archive_stream_provider, mime_type, password)
                if not archive:
                    raise IOError(f"Could not open archive for entity {target_entity_id}")

                member_stream = None
                try:
                    _, member_stream = archive.open_file(path_in_archive)
                    yield member_stream
                finally:
                    # Ensure the member stream is closed if it's not done by the consumer
                    if member_stream:
                        try:
                            member_stream.close()
                        except Exception:
                            pass
                    await archive.close()

        return ArchiveMemberStreamProvider()

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
    cmd: IngestArchiveCommand,
    archive_stream_provider: StreamProvider,
    world: Annotated[World, "Resource"],
    transaction: WorldTransaction,
) -> AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent, InformationRequest[Any]], Any]:
    """
    The core extraction and event-issuing logic for an archive.
    It performs the initial ingestion and handles encrypted archives by yielding PasswordRequests.
    """
    yield ProgressStarted()
    entity_id = cmd.entity_id

    mime_type = await get_content_mime_type(transaction.session, entity_id)
    if not mime_type:
        yield ProgressError(message=f"Could not get mime type for archive entity {entity_id}", exception=ValueError())
        return

    async def _try_open_archive(password: Optional[str]):
        try:
            return await open_archive(archive_stream_provider, mime_type, password)
        except InvalidPasswordError:
            return None
        except (IOError, RuntimeError) as e:
            return ProgressError(message=f"Failed to open archive {entity_id}", exception=e)
        except Exception:
            raise

    # --- Password Resolution ---
    archive = None
    correct_password = None
    stored_password_comp = await transaction.get_component(entity_id, ArchivePasswordComponent)

    # 1. Try passwords in a specific, non-interactive order.
    passwords_to_try = [cmd.password, stored_password_comp.password if stored_password_comp else None, None]
    # Create a list of unique passwords to try, preserving order. `None` is a valid entry.
    unique_passwords = list(dict.fromkeys(passwords_to_try))

    for pwd in unique_passwords:
        opened_archive = await _try_open_archive(pwd)
        if isinstance(opened_archive, ProgressError):
            yield opened_archive
            return
        if opened_archive:
            archive, correct_password = opened_archive, pwd
            logger.info(f"Successfully opened archive {entity_id} with password: {'yes' if pwd else 'no'}")
            break

    # 2. If still not open, enter the interactive loop.
    if not archive:
        is_first_request = True
        while True:
            message = "Invalid password." if not is_first_request else f"Password required for archive {entity_id}."
            new_password = yield PasswordRequest(message=message)
            is_first_request = False

            if new_password is None:
                yield ProgressError(
                    message=f"Password not provided for archive entity {entity_id}",
                    exception=PasswordRequiredError(),
                )
                return

            opened_archive = await _try_open_archive(new_password)
            if isinstance(opened_archive, ProgressError):
                yield opened_archive
                return
            if opened_archive:
                archive, correct_password = opened_archive, new_password
                logger.info(f"Successfully opened archive {entity_id} with provided password.")
                break
            else:
                logger.info(f"Invalid password provided for archive {entity_id}.")

    if not archive:
        return

    try:
        # --- Main Logic ---
        all_members = archive.list_files()
        logger.info(f"Entity {entity_id} is being ingested for the first time.")
        total_size = sum(m.size for m in all_members)
        yield ProgressUpdate(total=total_size, current=0, message="Starting ingestion.")

        processed_size = 0
        member_mod_times = {m.name: m.modified_at for m in all_members}

        for member_info, member_file in archive.iter_files():
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

                        event_stream_provider = CallableStreamProvider(event_provider)
                    else:
                        # For large files, we can't provide a re-readable event stream.
                        # The command gets a chained stream which consumes the buffer and the rest of the file stream.
                        in_memory_buffer.seek(0)
                        stream_for_command = cast(BinaryIO, ChainedStream([in_memory_buffer, member_stream]))

                    # --- Entity handling (Create vs. Re-issue) ---
                    # Initial ingestion: Get or create entity from stream
                    member_entity_id: Optional[int] = None
                    get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream=stream_for_command)
                    member_entity_tuple = await world.dispatch_command(get_or_create_cmd).get_one_value()
                    if member_entity_tuple:
                        member_entity, _ = member_entity_tuple
                        member_entity_id = member_entity.id

                    if not member_entity_id:
                        raise ValueError(f"Could not get or create entity for archive member '{member_info.name}'")

                    # Add ArchiveMemberComponent for new members first to avoid race conditions.
                    member_comp = ArchiveMemberComponent(
                        archive_entity_id=entity_id,
                        path_in_archive=member_info.name,
                        modified_at=member_mod_times.get(member_info.name),
                    )
                    await transaction.add_component_to_entity(member_entity_id, member_comp)

                    # --- Event Emission ---
                    if member_entity_id:
                        yield NewEntityCreatedEvent(
                            entity_id=member_entity_id,
                            stream_provider=event_stream_provider,
                            filename=member_info.name,
                        )

                # --- Progress Update ---
                processed_size += member_info.size
                yield ProgressUpdate(
                    total=total_size, current=processed_size, message=f"Processed '{member_info.name}'."
                )

            except Exception as e:
                logger.error(f"Failed to process member '{member_info.name}' from archive {entity_id}: {e}")
                # Don't abort the whole process for one bad file
                continue

        # --- Finalization ---
        # Save the correct password if it wasn't already stored
        if correct_password and (not stored_password_comp or correct_password != stored_password_comp.password):
            await world.dispatch_command(
                SetArchivePasswordCommand(entity_id=entity_id, password=correct_password)
            ).get_one_value()

        info_comp = ArchiveInfoComponent(comment=archive.comment)
        await transaction.add_or_update_component(entity_id, info_comp)
        logger.info(f"Finished processing archive {entity_id}, processed {len(all_members)} members.")
        yield ProgressCompleted()

    finally:
        if archive:
            await archive.close()


@system(on_command=IngestArchiveCommand)
async def ingest_archive_members_handler(
    cmd: IngestArchiveCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent, InformationRequest[Any]], Any]:
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

            class ChainedStreamProvider(StreamProvider):
                def __init__(self, providers: List[StreamProvider]):
                    self._providers = providers

                @asynccontextmanager
                async def get_stream(self) -> AsyncIterator[BinaryIO]:
                    streams: List[BinaryIO] = []
                    context_managers: List[AsyncContextManager[BinaryIO]] = []
                    try:
                        for p in self._providers:
                            cm = p.get_stream()
                            streams.append(await cm.__aenter__())
                            context_managers.append(cm)

                        chained_stream = cast(BinaryIO, ChainedStream(streams))
                        yield chained_stream
                    finally:
                        for cm in reversed(context_managers):
                            await cm.__aexit__(None, None, None)

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
                    all_providers = await world.dispatch_command(stream_cmd).get_all_results()
                    valid_providers = [p for p in all_providers if p is not None]
                    if valid_providers:
                        part_stream_providers.append(valid_providers[0])
                    else:
                        raise ValueError(f"Stream provider for part {part_entity_id} is None")

                archive_stream_provider = ChainedStreamProvider(part_stream_providers)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Could not get stream for split archive part: {e}")
                yield ProgressError(message=str(e), exception=e)
                return

        # Case 2: Part of an already assembled split archive.
        part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)
        if not manifest_comp and part_info and part_info.master_entity_id:
            logger.info(
                f"Entity {cmd.entity_id} is part of an assembled split archive (master: {part_info.master_entity_id}). "
                "Ingestion should be initiated from the master entity. Skipping ingestion for this part."
            )
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

    # The `_process_archive` generator can now make requests, so we need to
    # yield its events and pipe back any values sent to this generator.
    value_to_send = None
    process_gen = _process_archive(cmd, archive_stream_provider, world, transaction)
    try:
        while True:
            event = await process_gen.asend(value_to_send)
            value_to_send = yield event
    except StopAsyncIteration:
        pass


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
