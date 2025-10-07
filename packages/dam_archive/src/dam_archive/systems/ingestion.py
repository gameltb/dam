"""Systems for ingesting archives."""

import contextlib
import datetime
import io
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import (
    Annotated,
    Any,
    BinaryIO,
    TypedDict,
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

from ..base import ArchiveHandler, ArchiveMemberInfo
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
    """Check if the ArchiveInfoComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    return component is not None


@system(on_command=ReissueArchiveMemberEventsCommand)
async def reissue_archive_member_events_handler(
    cmd: ReissueArchiveMemberEventsCommand,
    transaction: WorldTransaction,
) -> AsyncGenerator[SystemProgressEvent | NewEntityCreatedEvent, None]:
    """Re-issue NewEntityCreatedEvent events for all members of an existing archive."""
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


def _create_archive_member_stream_provider(
    archive_stream_provider: StreamProvider,
    mime_type: str,
    password: str | None,
    target_entity_id: int,
    path_in_archive: str,
) -> StreamProvider:
    class ArchiveMemberStreamProvider(StreamProvider):
        @asynccontextmanager
        async def get_stream(self) -> AsyncIterator[BinaryIO]:
            archive = await open_archive(archive_stream_provider, mime_type, password)
            if not archive:
                raise OSError(f"Could not open archive for entity {target_entity_id}")

            member_stream = None
            try:
                _, member_stream = archive.open_file(path_in_archive)
                yield member_stream
            finally:
                # Ensure the member stream is closed if it's not done by the consumer
                if member_stream:
                    with contextlib.suppress(Exception):
                        member_stream.close()
                await archive.close()

    return ArchiveMemberStreamProvider()


def _create_command_stream_provider(stream: BinaryIO) -> CallableStreamProvider:
    """Create a stream provider for a command from a given stream."""
    return CallableStreamProvider(lambda: stream)


@system(on_command=GetAssetStreamCommand)
async def get_archive_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> StreamProvider | None:
    """Get a stream provider for an asset that is part of an archive."""
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
            logger.warning("Could not get stream provider for parent archive %s", target_entity_id)
            continue
        archive_stream_provider = valid_providers[0]

        if not archive_stream_provider:
            logger.warning("Could not get stream provider for parent archive %s", target_entity_id)
            continue

        mime_type = await get_content_mime_type(transaction.session, target_entity_id)
        if not mime_type:
            logger.warning("Could not get mime type for parent archive %s", target_entity_id)
            continue

        return _create_archive_member_stream_provider(
            archive_stream_provider,
            mime_type,
            password,
            target_entity_id,
            path_in_archive,
        )

    return None


@system(on_command=GetAssetFilenamesCommand)
async def get_archive_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: WorldTransaction,
) -> list[str] | None:
    """Get filenames for assets that are members of an archive."""
    archive_member_comps = await transaction.get_components(cmd.entity_id, ArchiveMemberComponent)
    if archive_member_comps:
        return [archive_member_comp.path_in_archive for archive_member_comp in archive_member_comps]
    return None


class ChainedStreamProvider(StreamProvider):
    """A stream provider that chains multiple stream providers together."""

    def __init__(self, providers: list[StreamProvider]):
        """
        Initialize the ChainedStreamProvider.

        Args:
            providers: A list of stream providers to chain.

        """
        self._providers = providers

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        """
        Get a chained stream from the providers.

        Yields:
            A chained binary stream.

        """
        streams: list[BinaryIO] = []
        context_managers: list[contextlib.AbstractAsyncContextManager[BinaryIO]] = []
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


async def _get_archive_stream_provider(  # noqa: PLR0911
    cmd: IngestArchiveCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> tuple[StreamProvider | None, ProgressError | None]:
    """Determine the correct StreamProvider for the given IngestArchiveCommand."""
    if cmd.stream_provider:
        logger.info("Processing archive for entity %s from provided stream provider.", cmd.entity_id)
        return cmd.stream_provider, None

    manifest_comp = await transaction.get_component(cmd.entity_id, SplitArchiveManifestComponent)
    part_info = await transaction.get_component(cmd.entity_id, SplitArchivePartInfoComponent)

    # Case 1: Master entity for a split archive.
    if manifest_comp:
        logger.info("Entity %s is a split archive master. Chaining part streams.", cmd.entity_id)
        stmt = (
            select(SplitArchivePartInfoComponent)
            .where(SplitArchivePartInfoComponent.master_entity_id == cmd.entity_id)
            .order_by(SplitArchivePartInfoComponent.part_num)
        )
        result = await transaction.session.execute(stmt)
        parts = result.scalars().all()
        part_entity_ids = [part.entity_id for part in parts]
        try:
            part_stream_providers: list[StreamProvider] = []
            for part_entity_id in part_entity_ids:
                stream_cmd = GetAssetStreamCommand(entity_id=part_entity_id)
                all_providers = await world.dispatch_command(stream_cmd).get_all_results()
                valid_providers = [p for p in all_providers if p is not None]
                if valid_providers:
                    part_stream_providers.append(valid_providers[0])
                else:
                    raise ValueError(f"Stream provider for part {part_entity_id} is None")

            return ChainedStreamProvider(part_stream_providers), None
        except (ValueError, FileNotFoundError) as e:
            logger.error("Could not get stream for split archive part: %s", e)
            return None, ProgressError(message=str(e), exception=e)

    # Case 2: Part of an already assembled split archive.
    if not manifest_comp and part_info and part_info.master_entity_id:
        logger.info(
            "Entity %s is part of an assembled split archive (master: %s). "
            "Ingestion should be initiated from the master entity. Skipping ingestion for this part.",
            cmd.entity_id,
            part_info.master_entity_id,
        )
        return None, None

    # Case 3: Part of a non-assembled split archive.
    if not manifest_comp and part_info and not part_info.master_entity_id:
        err_msg = (
            f"Entity {cmd.entity_id} is part of a non-assembled split archive. "
            "Please run 'discover-and-bind' or 'create-master' command first."
        )
        return None, ProgressError(message=err_msg, exception=RuntimeError(err_msg))

    # Case 4: Regular, single-file archive.
    if not manifest_comp and not part_info:
        logger.info("Entity %s is a single-file archive. Getting stream provider.", cmd.entity_id)
        try:
            stream_provider = await cmd.get_stream_provider(world)
            return stream_provider, None
        except (ValueError, FileNotFoundError) as e:
            logger.error("Could not get stream for single-file archive %s: %s", cmd.entity_id, e)
            return None, ProgressError(message=str(e), exception=e)

    return None, None  # Should not be reached, but mypy needs it.


async def _resolve_password_and_open_archive(
    cmd: IngestArchiveCommand,
    archive_stream_provider: StreamProvider,
    transaction: WorldTransaction,
    mime_type: str,
) -> AsyncGenerator[SystemProgressEvent | InformationRequest[Any] | tuple["ArchiveHandler", str | None], Any]:
    """
    Handle password resolution and opens the archive.

    This is an async generator that yields ProgressError or PasswordRequest events.
    If successful, it yields a tuple of (ArchiveHandler, correct_password) as its
    final event before stopping.
    """
    entity_id = cmd.entity_id

    async def _try_open_archive(password: str | None):
        try:
            return await open_archive(archive_stream_provider, mime_type, password)
        except InvalidPasswordError:
            return None
        except (OSError, RuntimeError) as e:
            return ProgressError(message=f"Failed to open archive {entity_id}", exception=e)
        except Exception:
            raise

    # --- Password Resolution ---
    stored_password_comp = await transaction.get_component(entity_id, ArchivePasswordComponent)
    passwords_to_try = [cmd.password, stored_password_comp.password if stored_password_comp else None, None]
    unique_passwords = list(dict.fromkeys(passwords_to_try))

    for pwd in unique_passwords:
        opened_archive = await _try_open_archive(pwd)
        if isinstance(opened_archive, ProgressError):
            yield opened_archive
            return
        if opened_archive:
            logger.info("Successfully opened archive %s with password: %s", entity_id, "yes" if pwd else "no")
            yield (opened_archive, pwd)
            return

    # Interactive loop if passwords failed
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
            logger.info("Successfully opened archive %s with provided password.", entity_id)
            yield (opened_archive, new_password)
            return
        else:
            logger.info("Invalid password provided for archive %s.", entity_id)


class MemberProcessingContext(TypedDict):
    """A TypedDict for the context of processing a member."""

    entity_id: int
    member_mod_times: dict[str, datetime.datetime | None]
    world: Annotated[World, "Resource"]
    transaction: WorldTransaction


async def _process_member(
    member_info: ArchiveMemberInfo,
    member_file: BinaryIO,
    context: MemberProcessingContext,
) -> AsyncGenerator[NewEntityCreatedEvent, None]:
    """Process a single member of an archive, yielding a NewEntityCreatedEvent on success."""
    world = context["world"]
    transaction = context["transaction"]
    entity_id = context["entity_id"]
    member_mod_times = context["member_mod_times"]

    with member_file as member_stream:
        # --- Memory-constrained stream handling ---
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * 0.5)
        in_memory_buffer = io.BytesIO(member_stream.read(memory_limit))
        is_eof = not member_stream.read(1)

        event_stream_provider: StreamProvider | None = None
        if is_eof:
            in_memory_buffer.seek(0)
            buffer_content = in_memory_buffer.read()

            def event_provider(content: bytes = buffer_content) -> BinaryIO:
                return io.BytesIO(content)

            event_stream_provider = CallableStreamProvider(event_provider)
            command_stream_provider = event_stream_provider
        else:
            in_memory_buffer.seek(0)
            stream_for_command = cast(BinaryIO, ChainedStream([in_memory_buffer, member_stream]))
            command_stream_provider = _create_command_stream_provider(stream_for_command)

        # --- Entity handling ---
        get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream_provider=command_stream_provider)
        member_entity_tuple = await world.dispatch_command(get_or_create_cmd).get_one_value()
        if not member_entity_tuple:
            raise ValueError(f"Could not get or create entity for archive member '{member_info.name}'")

        member_entity, _ = member_entity_tuple
        member_entity_id = member_entity.id

        # Add ArchiveMemberComponent
        member_comp = ArchiveMemberComponent(
            archive_entity_id=entity_id,
            path_in_archive=member_info.name,
            modified_at=member_mod_times.get(member_info.name),
        )
        await transaction.add_component_to_entity(member_entity_id, member_comp)

        # --- Event Emission ---
        yield NewEntityCreatedEvent(
            entity_id=member_entity_id,
            stream_provider=event_stream_provider,
            filename=member_info.name,
        )


async def _process_archive(
    cmd: IngestArchiveCommand,
    archive_stream_provider: StreamProvider,
    world: Annotated[World, "Resource"],
    transaction: WorldTransaction,
) -> AsyncGenerator[SystemProgressEvent | NewEntityCreatedEvent | InformationRequest[Any], Any]:
    """Perform the core extraction and event-issuing logic for an archive."""
    yield ProgressStarted()
    entity_id = cmd.entity_id

    mime_type = await get_content_mime_type(transaction.session, entity_id)
    if not mime_type:
        yield ProgressError(message=f"Could not get mime type for archive entity {entity_id}", exception=ValueError())
        return

    # --- Password Resolution ---
    archive: ArchiveHandler | None = None
    correct_password: str | None = None
    password_gen = _resolve_password_and_open_archive(cmd, archive_stream_provider, transaction, mime_type)
    response = None
    try:
        while True:
            event = await password_gen.asend(response)
            if isinstance(event, tuple):
                archive, correct_password = event
                break
            response = yield event
    except StopAsyncIteration:
        pass

    if not archive:
        return

    try:
        # --- Main Logic ---
        all_members = archive.list_files()
        logger.info("Entity %s is being ingested for the first time.", entity_id)
        total_size = sum(m.size for m in all_members)
        yield ProgressUpdate(total=total_size, current=0, message="Starting ingestion.")

        processed_size = 0
        context: MemberProcessingContext = {
            "entity_id": entity_id,
            "member_mod_times": {m.name: m.modified_at for m in all_members},
            "world": world,
            "transaction": transaction,
        }

        for member_info, member_file in archive.iter_files():
            try:
                async for event in _process_member(member_info, member_file, context):
                    yield event
            except Exception as e:
                logger.error("Failed to process member '%s' from archive %s: %s", member_info.name, entity_id, e)
            finally:
                processed_size += member_info.size
                yield ProgressUpdate(
                    total=total_size, current=processed_size, message=f"Processed '{member_info.name}'."
                )

        # --- Finalization ---
        stored_password_comp = await transaction.get_component(entity_id, ArchivePasswordComponent)
        if correct_password and (not stored_password_comp or correct_password != stored_password_comp.password):
            await world.dispatch_command(
                SetArchivePasswordCommand(entity_id=entity_id, password=correct_password)
            ).get_one_value()

        info_comp = ArchiveInfoComponent(comment=archive.comment)
        await transaction.add_or_update_component(entity_id, info_comp)
        logger.info("Finished processing archive %s, processed %s members.", entity_id, len(all_members))
        yield ProgressCompleted()

    finally:
        if archive:
            await archive.close()


@system(on_command=IngestArchiveCommand)
async def ingest_archive_members_handler(
    cmd: IngestArchiveCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> AsyncGenerator[SystemProgressEvent | NewEntityCreatedEvent | InformationRequest[Any], Any]:
    """
    Handle processing an archive. It's the main entry point for ingestion.

    This handler determines the stream provider and then calls the main processing logic.
    """
    logger.info("Ingestion command received for entity %s", cmd.entity_id)

    archive_stream_provider, error = await _get_archive_stream_provider(cmd, transaction, world)

    if error:
        yield error
        return

    if not archive_stream_provider:
        # This can happen if the file is a part of an already assembled archive, which is not an error.
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
    """Clear archive-related components from an entity and its members."""
    # Delete ArchiveInfoComponent from the main archive entity
    info_comp = await transaction.get_component(cmd.entity_id, ArchiveInfoComponent)
    if info_comp:
        await transaction.remove_component(info_comp)
        logger.info("Deleted ArchiveInfoComponent from entity %s", cmd.entity_id)

    # Find all ArchiveMemberComponents that point to this archive entity
    stmt = select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    member_components = result.scalars().all()

    for member_comp in member_components:
        await transaction.remove_component(member_comp)
        logger.info(
            "Deleted ArchiveMemberComponent from member entity %s (linked to archive %s)",
            member_comp.entity_id,
            cmd.entity_id,
        )
