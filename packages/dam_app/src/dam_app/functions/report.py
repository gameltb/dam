"""Functions for generating reports."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url
from dam_psp.models import CsoParentIsoComponent
from sqlalchemy import String, and_, func, literal, or_, select, union_all
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

DuplicateRow = Row[tuple[int, int | None, int | None, bytes, int, str, str]]
EntryRow = Any


@dataclass
class DeletePlanRow:
    """Represents a row in the delete plan."""

    source_path: str
    target_path: str
    hash: str
    size: int
    details: str


async def create_delete_plan(  # noqa: PLR0912
    session: AsyncSession,
    min_size_bytes: int,
    keep_dirs: Sequence[Path] | None = None,
    delete_dirs: Sequence[Path] | None = None,
) -> Sequence[DeletePlanRow]:
    """Create a plan to delete duplicate files from the target directory."""

    def get_all_entries_query():
        """Create a query to get all file-like entries (direct files and archive members)."""
        direct_loc = aliased(FileLocationComponent)
        direct_locations_query = (
            select(
                ContentHashSHA256Component.hash_value,
                ContentLengthComponent.file_size_bytes,
                direct_loc.url.label("path"),
                literal(None, type_=String).label("member_path"),
            )
            .select_from(direct_loc)
            .join(
                ContentHashSHA256Component,
                ContentHashSHA256Component.entity_id == direct_loc.entity_id,
            )
            .outerjoin(
                ContentLengthComponent,
                ContentLengthComponent.entity_id == direct_loc.entity_id,
            )
        )

        archive_loc = aliased(FileLocationComponent)
        archive_locations_query = (
            select(
                ContentHashSHA256Component.hash_value,
                ContentLengthComponent.file_size_bytes,
                archive_loc.url.label("path"),
                ArchiveMemberComponent.path_in_archive.label("member_path"),
            )
            .select_from(ArchiveMemberComponent)
            .join(
                ContentHashSHA256Component,
                ContentHashSHA256Component.entity_id == ArchiveMemberComponent.entity_id,
            )
            .join(
                archive_loc,
                ArchiveMemberComponent.archive_entity_id == archive_loc.entity_id,
            )
            .outerjoin(
                ContentLengthComponent,
                ContentLengthComponent.entity_id == ArchiveMemberComponent.entity_id,
            )
        )
        return union_all(direct_locations_query, archive_locations_query)

    def format_path(row: EntryRow) -> str:
        path: str = str(get_local_path_for_url(row.path))
        if row.member_path:
            return f"{path} -> {row.member_path}"
        return path

    all_entries_query = get_all_entries_query()
    all_results = (await session.execute(all_entries_query)).all()

    entries_by_hash: dict[bytes, list[EntryRow]] = {}
    for row in all_results:
        entries_by_hash.setdefault(row.hash_value, []).append(row)

    delete_list: list[DeletePlanRow] = []

    for hash_value, entries in entries_by_hash.items():
        if len(entries) < 2:  # noqa: PLR2004
            continue

        keep_list: list[EntryRow] = []
        candidates: list[EntryRow] = []

        keep_urls = [d.as_uri() for d in keep_dirs] if keep_dirs else []
        delete_urls = [d.as_uri() for d in delete_dirs] if delete_dirs else []

        for entry in entries:
            if any(entry.path.startswith(url) for url in keep_urls):
                keep_list.append(entry)
            elif any(entry.path.startswith(url) for url in delete_urls):
                candidates.append(entry)
            else:
                candidates.append(entry)

        if not keep_list:
            keep_list.append(candidates.pop(0))

        for item_to_delete in candidates:
            source_item = keep_list[0]
            source_path_str = format_path(source_item)
            target_path_str = format_path(item_to_delete)

            delete_list.append(
                DeletePlanRow(
                    source_path=source_path_str,
                    target_path=target_path_str,
                    hash=hash_value.hex(),
                    size=item_to_delete.file_size_bytes or 0,
                    details=f"Duplicate of {source_path_str}",
                )
            )

    # Group archive members by archive path
    archives_to_process: dict[str, list[DeletePlanRow]] = {}
    final_delete_plan: list[DeletePlanRow] = []
    for item in delete_list:
        if " -> " in item.target_path:
            archive_path = item.target_path.split(" -> ")[0]
            archives_to_process.setdefault(archive_path, []).append(item)
        else:
            final_delete_plan.append(item)

    all_entries = get_all_entries_query().subquery()
    for archive_path_str, members_to_delete in archives_to_process.items():
        archive_path_url = Path(archive_path_str).as_uri()
        all_members_in_archive_query = select(all_entries).where(
            and_(all_entries.c.path == archive_path_url, all_entries.c.member_path.is_not(None))
        )
        all_members = (await session.execute(all_members_in_archive_query)).all()

        is_fully_duplicated = len(all_members) == len(members_to_delete)
        total_duplicated_size = sum(m.size for m in members_to_delete)

        if is_fully_duplicated or (members_to_delete and total_duplicated_size >= min_size_bytes):
            archive_file_info_query = (
                select(
                    ContentHashSHA256Component.hash_value,
                    ContentLengthComponent.file_size_bytes,
                )
                .join(FileLocationComponent, FileLocationComponent.entity_id == ContentHashSHA256Component.entity_id)
                .outerjoin(
                    ContentLengthComponent,
                    ContentLengthComponent.entity_id == ContentHashSHA256Component.entity_id,
                )
                .where(FileLocationComponent.url == archive_path_url)
            )
            archive_info = (await session.execute(archive_file_info_query)).one_or_none()

            if archive_info:
                details_list = [f"'{m.target_path}' is a duplicate of '{m.source_path}'" for m in members_to_delete]
                details_str = "; ".join(details_list)

                details = (
                    f"All members are duplicates: [{details_str}]"
                    if is_fully_duplicated
                    else f"Duplicate members size ({total_duplicated_size} bytes) exceeds threshold: [{details_str}]"
                )
                final_delete_plan.append(
                    DeletePlanRow(
                        source_path="Multiple files",
                        target_path=archive_path_str,
                        hash=archive_info.hash_value.hex(),
                        size=archive_info.file_size_bytes or 0,
                        details=details,
                    )
                )

    return final_delete_plan


async def get_duplicates_report(session: AsyncSession, path: Path | None = None) -> Sequence[DuplicateRow]:
    """Get duplicate files from the database."""
    # Subquery to count the number of locations for each entity
    location_counts_subquery = (
        select(
            Entity.id.label("entity_id"),
            func.count(FileLocationComponent.id).label("file_location_count"),
            func.count(ArchiveMemberComponent.id).label("archive_member_count"),
            func.count(CsoParentIsoComponent.id).label("cso_parent_iso_count"),
        )
        .select_from(Entity)
        .outerjoin(FileLocationComponent, Entity.id == FileLocationComponent.entity_id)
        .outerjoin(ArchiveMemberComponent, Entity.id == ArchiveMemberComponent.entity_id)
        .outerjoin(CsoParentIsoComponent, Entity.id == CsoParentIsoComponent.entity_id)
        .group_by(Entity.id)
        .subquery()
    )

    # Alias for FileLocationComponent to distinguish between direct and indirect locations
    direct_loc = aliased(FileLocationComponent)
    archive_loc = aliased(FileLocationComponent)
    cso_loc = aliased(FileLocationComponent)
    container_content_length = aliased(ContentLengthComponent)

    # Query for direct file locations
    direct_locations_query = (
        select(
            Entity.id.label("entity_id"),
            ContentLengthComponent.file_size_bytes,
            ContentLengthComponent.file_size_bytes.label("size_on_disk"),
            ContentHashSHA256Component.hash_value,
            (
                location_counts_subquery.c.file_location_count
                + location_counts_subquery.c.archive_member_count
                + location_counts_subquery.c.cso_parent_iso_count
            ).label("total_locations"),
            direct_loc.url.label("path"),
            literal("Filesystem").label("type"),
        )
        .select_from(Entity)
        .join(location_counts_subquery, location_counts_subquery.c.entity_id == Entity.id)
        .join(direct_loc, Entity.id == direct_loc.entity_id)
        .outerjoin(ContentLengthComponent, ContentLengthComponent.entity_id == Entity.id)
        .join(ContentHashSHA256Component, ContentHashSHA256Component.entity_id == Entity.id)
    )

    # Query for archive member locations
    archive_locations_query = (
        select(
            Entity.id.label("entity_id"),
            ContentLengthComponent.file_size_bytes,
            ArchiveMemberComponent.compressed_size.label("size_on_disk"),
            ContentHashSHA256Component.hash_value,
            (
                location_counts_subquery.c.file_location_count
                + location_counts_subquery.c.archive_member_count
                + location_counts_subquery.c.cso_parent_iso_count
            ).label("total_locations"),
            (archive_loc.url.cast(String) + " -> " + ArchiveMemberComponent.path_in_archive).label("path"),
            literal("Archive").label("type"),
        )
        .select_from(Entity)
        .join(location_counts_subquery, location_counts_subquery.c.entity_id == Entity.id)
        .join(ArchiveMemberComponent, Entity.id == ArchiveMemberComponent.entity_id)
        .join(archive_loc, ArchiveMemberComponent.archive_entity_id == archive_loc.entity_id)
        .outerjoin(ContentLengthComponent, ContentLengthComponent.entity_id == Entity.id)
        .outerjoin(
            container_content_length, container_content_length.entity_id == ArchiveMemberComponent.archive_entity_id
        )
        .join(ContentHashSHA256Component, ContentHashSHA256Component.entity_id == Entity.id)
    )

    # Query for CSO parent locations
    cso_locations_query = (
        select(
            Entity.id.label("entity_id"),
            ContentLengthComponent.file_size_bytes,
            container_content_length.file_size_bytes.label("size_on_disk"),
            ContentHashSHA256Component.hash_value,
            (
                location_counts_subquery.c.file_location_count
                + location_counts_subquery.c.archive_member_count
                + location_counts_subquery.c.cso_parent_iso_count
            ).label("total_locations"),
            cso_loc.url.label("path"),
            literal("CSO").label("type"),
        )
        .select_from(Entity)
        .join(location_counts_subquery, location_counts_subquery.c.entity_id == Entity.id)
        .join(CsoParentIsoComponent, Entity.id == CsoParentIsoComponent.entity_id)
        .join(cso_loc, CsoParentIsoComponent.cso_entity_id == cso_loc.entity_id)
        .outerjoin(ContentLengthComponent, ContentLengthComponent.entity_id == Entity.id)
        .outerjoin(container_content_length, container_content_length.entity_id == CsoParentIsoComponent.cso_entity_id)
        .join(ContentHashSHA256Component, ContentHashSHA256Component.entity_id == Entity.id)
    )

    # Combine all locations with UNION
    all_locations_query = union_all(
        direct_locations_query,
        archive_locations_query,
        cso_locations_query,
    ).subquery()

    # Final query to select duplicates and filter by path if provided
    duplicates_query = select(all_locations_query).where(all_locations_query.c.total_locations > 1)

    if path:
        path_filter = f"file://{path}"
        duplicates_query = duplicates_query.where(
            or_(
                all_locations_query.c.path.startswith(path_filter),
                and_(
                    all_locations_query.c.type == "Archive",
                    all_locations_query.c.path.startswith(path_filter),
                ),
            )
        )

    result = await session.execute(duplicates_query)
    return result.all()
