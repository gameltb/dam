from dam.models.core import BaseComponent as Component
from sqlalchemy import Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveInfoComponent(Component):
    """
    A component that marks an asset as an archive that has been processed.
    It can store basic metadata about the archive itself.
    """

    __tablename__ = "component_archive_info"

    file_count: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", name="uq_archive_info_entity"),)

    def __repr__(self) -> str:
        return f"ArchiveInfoComponent(id={self.id}, entity_id={self.entity_id}, file_count={self.file_count})"
