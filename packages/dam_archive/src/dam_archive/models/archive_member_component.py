from datetime import datetime
from typing import Optional

from dam.models.core import BaseComponent as Component
from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveMemberComponent(Component):
    """
    A component that marks an asset as a member of an archive.
    """

    __tablename__ = "component_archive_member"

    archive_entity_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("entities.id"), nullable=False, index=True)
    path_in_archive: Mapped[str] = mapped_column(String(), nullable=False)
    modified_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
