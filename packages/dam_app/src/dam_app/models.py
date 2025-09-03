from dam.models.core import BaseComponent as Component
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveMemberComponent(Component):
    """
    A component that marks an asset as a member of an archive.
    """

    __tablename__ = "component_archive_member"

    archive_entity_id: Mapped[int] = mapped_column(nullable=False)
    path_in_archive: Mapped[str] = mapped_column(String(4096), nullable=False)
