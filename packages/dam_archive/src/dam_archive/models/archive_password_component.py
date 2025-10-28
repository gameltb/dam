"""Defines the ArchivePasswordComponent."""

from dam.models.core import BaseComponent as Component
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class ArchivePasswordComponent(Component):
    """Stores the password for an encrypted archive."""

    __tablename__ = "component_archive_password"

    password: Mapped[str] = mapped_column(String(), nullable=False)

    __table_args__ = (UniqueConstraint("entity_id", name="uq_password_entity"),)
