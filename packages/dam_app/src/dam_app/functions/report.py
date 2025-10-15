"""Functions for generating reports."""

from collections.abc import Sequence
from pathlib import Path

from dam.models.core import Entity
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_psp.models import CsoParentIsoComponent
from sqlalchemy import func, select
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession

DuplicateRow = Row[tuple[int, int, int | None, bytes]]


async def get_duplicates_report(session: AsyncSession, path: Path | None = None) -> Sequence[DuplicateRow]:
    """Get duplicate files from the database."""
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
        .select_from(Entity)
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

    if path:
        path_filter_subquery = select(FileLocationComponent.entity_id).where(
            FileLocationComponent.url.startswith(f"file://{path}")
        )
        duplicates_query = duplicates_query.where(location_counts_subquery.c.entity_id.in_(path_filter_subquery))

    result = await session.execute(duplicates_query)
    return result.all()
