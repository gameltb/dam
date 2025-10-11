"""CLI commands for generating reports."""

from dam.core.transaction import WorldTransaction
from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_psp.models import CsoParentIsoComponent
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="report",
    help="Commands for generating reports.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


@app.command("duplicates")
async def report_duplicates():
    """Report on duplicate files in the world."""
    world = get_world()
    if not world:
        return

    console = Console()

    async with world.get_context(WorldTransaction)() as transaction:
        session = transaction.session

        location_counts_subquery = (
            select(
                Entity.id.label("entity_id"),
                func.count(FileLocationComponent.id).label("file_location_count"),
                func.count(ArchiveMemberComponent.id).label("archive_member_count"),
                func.count(CsoParentIsoComponent.id).label("cso_parent_iso_count"),
            )
            .outerjoin(FileLocationComponent, Entity.id == FileLocationComponent.entity_id)
            .outerjoin(ArchiveMemberComponent, Entity.id == ArchiveMemberComponent.entity_id)
            .outerjoin(CsoParentIsoComponent, Entity.id == CsoParentIsoComponent.entity_id)
            .group_by(Entity.id)
            .subquery()
        )

        duplicates_query = (
            select(
                location_counts_subquery.c.entity_id,
                (
                    location_counts_subquery.c.file_location_count
                    + location_counts_subquery.c.archive_member_count
                    + location_counts_subquery.c.cso_parent_iso_count
                ).label("total_locations"),
                ContentLengthComponent.file_size_bytes,
                ContentHashSHA256Component.hash_value,
            )
            .join(location_counts_subquery, location_counts_subquery.c.entity_id == Entity.id)
            .outerjoin(ContentLengthComponent, ContentLengthComponent.entity_id == Entity.id)
            .join(ContentHashSHA256Component, ContentHashSHA256Component.entity_id == Entity.id)
            .where(
                (
                    location_counts_subquery.c.file_location_count
                    + location_counts_subquery.c.archive_member_count
                    + location_counts_subquery.c.cso_parent_iso_count
                )
                > 1
            )
        )

        result = await session.execute(duplicates_query)
        duplicates = result.all()

        if not duplicates:
            console.print("No duplicate files found.")
            return

        total_wasted_space = 0
        table = Table(title="Duplicate Files Report")
        table.add_column("Entity ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("SHA256", style="magenta")
        table.add_column("Size (bytes)", justify="right", style="green")
        table.add_column("Locations", justify="right", style="red")
        table.add_column("Wasted Space (bytes)", justify="right", style="yellow")
        table.add_column("Paths", style="blue")

        for duplicate in duplicates:
            entity_id = duplicate.entity_id
            total_locations = duplicate.total_locations
            size_bytes = duplicate.file_size_bytes or 0
            hash_hex = duplicate.hash_value.hex()

            wasted_space = size_bytes * (total_locations - 1)
            total_wasted_space += wasted_space

            paths: list[str] = []
            file_locations_result = await session.execute(
                select(FileLocationComponent).where(FileLocationComponent.entity_id == entity_id)
            )
            file_locations = file_locations_result.scalars().all()
            archive_members_result = await session.execute(
                select(ArchiveMemberComponent).where(ArchiveMemberComponent.entity_id == entity_id)
            )
            archive_members = archive_members_result.scalars().all()
            cso_parents_result = await session.execute(
                select(CsoParentIsoComponent).where(CsoParentIsoComponent.entity_id == entity_id)
            )
            cso_parents = cso_parents_result.scalars().all()

            for loc in file_locations:
                paths.append(f"Filesystem: {loc.url}")

            for member in archive_members:
                archive_file_locations_result = await session.execute(
                    select(FileLocationComponent).where(FileLocationComponent.entity_id == member.archive_entity_id)
                )
                archive_file_locations = archive_file_locations_result.scalars().all()
                for archive_loc in archive_file_locations:
                    paths.append(f"Archive: {archive_loc.url} -> {member.path_in_archive}")

            for cso in cso_parents:
                cso_file_locations_result = await session.execute(
                    select(FileLocationComponent).where(FileLocationComponent.entity_id == cso.cso_entity_id)
                )
                cso_file_locations = cso_file_locations_result.scalars().all()
                for cso_loc in cso_file_locations:
                    paths.append(f"CSO: {cso_loc.url}")

            table.add_row(
                str(entity_id),
                f"{hash_hex[:16]}...",
                str(size_bytes),
                str(total_locations),
                str(wasted_space),
                "\n".join(paths),
            )

        console.print(table)
        console.print(f"\nTotal wasted space: {total_wasted_space} bytes")
