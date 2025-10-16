"""Temporary CLI commands."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dam.core.transaction import WorldTransaction
from dam.core.types import FileStreamProvider
from dam.models.core import Entity
from dam_archive.models import ArchiveMemberComponent
from dam_archive.registry import MIME_TYPE_HANDLERS
from dam_fs.functions.file_operations import get_mime_type_async
from dam_fs.models.file_location_component import FileLocationComponent
from rich.console import Console
from sqlalchemy import select

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="temp",
    help="Temporary commands.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


@app.command("backfill-compressed-size")
async def backfill_compressed_size(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="If set, the command will only print the changes that would be made.",
        ),
    ] = False,
):
    """Backfill the compressed_size field for ArchiveMemberComponent."""
    world = get_world()
    if not world:
        return

    console = Console()
    console.print("Starting backfill of compressed_size for ArchiveMemberComponent...")

    async with world.get_context(WorldTransaction)() as transaction:
        session = transaction.session
        query = select(ArchiveMemberComponent)
        result = await session.execute(query)
        members: Sequence[ArchiveMemberComponent] = result.scalars().all()

        members_by_archive: defaultdict[int, list[ArchiveMemberComponent]] = defaultdict(list)
        for member in members:
            members_by_archive[member.archive_entity_id].append(member)

        for archive_entity_id, archive_members in members_by_archive.items():
            archive_entity = await session.get(Entity, archive_entity_id)
            if not archive_entity:
                continue

            file_location_query = select(FileLocationComponent).where(
                FileLocationComponent.entity_id == archive_entity_id
            )
            file_location_result = await session.execute(file_location_query)
            file_location = file_location_result.scalars().first()

            if not file_location:
                continue

            filepath = Path(file_location.url.replace("file://", ""))
            mime_type = await get_mime_type_async(filepath)
            if mime_type not in MIME_TYPE_HANDLERS:
                continue

            handler_class = MIME_TYPE_HANDLERS[mime_type][0]
            stream_provider = FileStreamProvider(filepath)
            handler = await handler_class.create(stream_provider)

            files = handler.list_files()
            file_info_map = {file.name: file for file in files}

            for member in archive_members:
                if member.path_in_archive in file_info_map:
                    file_info = file_info_map[member.path_in_archive]
                    if dry_run:
                        console.print(
                            f"Would update member {member.id} with compressed_size: {file_info.compressed_size}"
                        )
                    else:
                        member.compressed_size = file_info.compressed_size
                        session.add(member)
                        console.print(f"Updated member {member.id} with compressed_size: {file_info.compressed_size}")

        if not dry_run:
            await session.commit()

    console.print("Backfill complete.")
