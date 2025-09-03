from dam.models.core import BaseComponent as Component
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class ArchiveMemberComponent(Component):
    """
    A component that marks an asset as a member of an archive.
    """

    __tablename__ = "component_archive_member"

    archive_entity_id: Mapped[int] = mapped_column(nullable=False)
    path_in_archive: Mapped[str] = mapped_column(String(4096), nullable=False)


class ArchivePasswordComponent(Component):
    """
    Stores the password for an encrypted archive.
    """

    __tablename__ = "component_archive_password"

    password: Mapped[str] = mapped_column(String(1024), nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", name="uq_password_entity"),)

    def __repr__(self):
        return f"ArchivePasswordComponent(id={self.id}, entity_id={self.entity_id})"
