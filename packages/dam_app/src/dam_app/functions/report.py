"""Functions for generating reports."""

from collections.abc import Sequence
from pathlib import Path

from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_psp.models import CsoParentIsoComponent
from sqlalchemy import String, and_, func, literal, or_, select, union_all
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

DuplicateRow = Row[tuple[int, int | None, int | None, bytes, int, str, str]]


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
