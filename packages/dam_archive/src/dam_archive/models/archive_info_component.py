from typing import Optional

from dam.models.core import UniqueComponent as Component
from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveInfoComponent(Component):
    """
    A component that marks an asset as an archive that has been processed.
    It can store basic metadata about the archive itself.
    """

    __tablename__ = "component_archive_info"

    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"ArchiveInfoComponent(entity_id={self.entity_id}, comment={self.comment})"
