from typing import Dict, Optional

from sqlalchemy import JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class WebsiteProfileComponent(BaseComponent):
    """
    Component attached to an Entity representing a website, storing common
    profile information and configurations for that website.
    """

    __tablename__ = "component_website_profile"

    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        comment="Unique human-readable name of the website (e.g., DeviantArt, ArtStation).",
    )
    main_url: Mapped[str] = mapped_column(
        String(2048), unique=True, index=True, nullable=False, comment="Main homepage URL of the website."
    )

    description: Mapped[Optional[str]] = mapped_column(Text, comment="Optional description of the website.", default=None)
    icon_url: Mapped[Optional[str]] = mapped_column(String(2048), comment="URL to a favicon or logo for the website.", default=None)

    # For potential API interactions or specific scraping configurations
    api_endpoint: Mapped[Optional[str]] = mapped_column(
        String(2048), comment="Primary API endpoint for the website, if applicable.", default=None
    )
    parser_rules: Mapped[Optional[Dict[str, any]]] = mapped_column(
        JSON,  # Using standard JSON for compatibility (e.g., SQLite)
        nullable=True,
        comment="JSON field to store site-specific parsing/scraping hints or configurations.",
        default=None # For dataclass __init__
    )

    # An entity with a WebsiteProfileComponent is considered a "Website Entity".
    # The entity_id (from BaseComponent) links this profile to that Website Entity.
    # Unique constraint on entity_id ensures one profile per website entity.
    __table_args__ = (UniqueConstraint("entity_id", name="uq_website_profile_entity_id"),)

    def __repr__(self):
        return (
            f"WebsiteProfileComponent(id={self.id}, entity_id={self.entity_id}, "
            f"name='{self.name}', main_url='{self.main_url}')"
        )
