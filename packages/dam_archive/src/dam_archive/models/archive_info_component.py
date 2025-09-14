from dam.models.core import BaseComponent as Component
from dam.models.core.component_mixins import UniqueComponentMixin
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveInfoComponent(UniqueComponentMixin, Component):
    """
    A component that marks an asset as an archive that has been processed.
    It can store basic metadata about the archive itself.
    """

    __tablename__ = "component_archive_info"

    file_count: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"ArchiveInfoComponent(id={self.id}, entity_id={self.entity_id}, file_count={self.file_count})"
