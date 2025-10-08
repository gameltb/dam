"""Defines the WebsiteProfileComponent model."""

from typing import Any

from dam.models.core import BaseComponent
from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class WebsiteProfileComponent(BaseComponent):
    """
    Component attached to an Entity representing a website.

    This component stores common profile information and configurations for that website.
    """

    __tablename__ = "component_website_profile"

    name: Mapped[str] = mapped_column(
        String(),
        unique=True,
        index=True,
        nullable=False,
        comment="Unique human-readable name of the website (e.g., DeviantArt, ArtStation).",
    )
    main_url: Mapped[str] = mapped_column(
        String(), unique=True, index=True, nullable=False, comment="Main homepage URL of the website."
    )

    description: Mapped[str | None] = mapped_column(Text, comment="Optional description of the website.", default=None)
    icon_url: Mapped[str | None] = mapped_column(
        String(), comment="URL to a favicon or logo for the website.", default=None
    )

    # For potential API interactions or specific scraping configurations
    api_endpoint: Mapped[str | None] = mapped_column(
        String(), comment="Primary API endpoint for the website, if applicable.", default=None
    )
    parser_rules: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="JSON field to store site-specific parsing/scraping hints or configurations.",
        default=None,  # For dataclass __init__
    )

    # An entity with a WebsiteProfileComponent is considered a "Website Entity".
    # The entity_id (from BaseComponent) links this profile to that Website Entity.
    # Unique constraint on entity_id ensures one profile per website entity.
    __table_args__ = (UniqueConstraint("entity_id", name="uq_website_profile_entity_id"),)

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return f"WebsiteProfileComponent(id={self.id}, entity_id={self.entity_id}, name='{self.name}', main_url='{self.main_url}')"
