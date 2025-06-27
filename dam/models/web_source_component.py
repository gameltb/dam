from typing import Optional, Dict
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, Integer # Added ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship # Added relationship
from datetime import datetime

from .base_component import BaseComponent
# from .types import JSONBType # Removed unused import, using sqlalchemy.JSON directly
from .entity import Entity # For relationship type hint


class WebSourceComponent(BaseComponent):
    """
    Component to store information about assets originating from web sources,
    linking to a Website Entity and detailing the specific asset's context on that site.
    """
    __tablename__ = "component_web_source"

    # Link to the Entity representing the website
    website_entity_id: Mapped[int] = mapped_column(
        Integer, # Explicitly using Integer, though ForeignKey handles type resolution too
        ForeignKey('entities.id', name='fk_web_source_website_entity_id'),
        index=True,
        nullable=False,
        comment="ID of the Entity that represents the source website."
    )
    website: Mapped["Entity"] = relationship(
        foreign_keys=[website_entity_id],
        backref="sourced_assets", # Or a more specific backref if needed
        doc="The Entity representing the website this asset came from."
    )

    source_url: Mapped[str] = mapped_column(String(2048), index=True, nullable=False,
                                          comment="URL of the asset's page or where it was found. Should be unique per (website_entity_id, gallery_id) or similar.")
    original_file_url: Mapped[Optional[str]] = mapped_column(String(2048),
                                                             comment="Direct URL to the media file, if different from source_url.")

    # website_name is now part of the linked Website Entity via WebsiteProfileComponent
    # gallery_id remains to identify the specific item on the site
    gallery_id: Mapped[Optional[str]] = mapped_column(String(255), index=True,
                                                      comment="Identifier for the gallery, post, or collection on the website (e.g., submission ID).")

    uploader_name: Mapped[Optional[str]] = mapped_column(String(255), index=True, # This might also become a link to a User/Artist Entity later
                                                         comment="Username of the uploader or artist on the site.")
    uploader_url: Mapped[Optional[str]] = mapped_column(String(2048),
                                                        comment="URL to the uploader's profile page.")

    upload_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True,
                                                            comment="Date and time when the asset was originally uploaded or posted.")

    asset_title: Mapped[Optional[str]] = mapped_column(String(1024), comment="Title of the asset on the website.")
    asset_description: Mapped[Optional[str]] = mapped_column(Text, comment="Description or caption provided for the asset.")

    # For tags, a simple text field for comma-separated or JSON string.
    # A dedicated Tagging system/component would be better for advanced tag management.
    tags_json: Mapped[Optional[str]] = mapped_column(Text, comment="JSON string or comma-separated list of tags from the source.")

    # For extensibility with gallery-specific metadata.
    # Using JSON type if database supports it (like PostgreSQL).
    # For SQLite, this might default to TEXT and require manual JSON parsing.
    # Using sqlalchemy.types.JSON for broader compatibility including SQLite.
    raw_metadata_dump: Mapped[Optional[Dict[str, any]]] = mapped_column(JSON, nullable=True,
                                                                       comment="Dump of raw metadata from the web source, for extensibility.")

    # Consider adding a 'downloaded_at' timestamp if the file is fetched later.
    # Consider a 'last_checked_at' timestamp for link validation.

    def __repr__(self):
        return (
            f"WebSourceComponent(id={self.id}, entity_id={self.entity_id}, website_entity_id={self.website_entity_id}, "
            f"source_url='{self.source_url[:50]}...')"
        )

    # Potential __table_args__:
    # Could add a UniqueConstraint on (entity_id) if an entity can only have one web source.
    # Or UniqueConstraint on (source_url) if source_urls should be unique across all web-sourced components (might be too restrictive).
    # For now, allowing multiple WebSourceComponents per entity, or relying on service layer logic.
    # __table_args__ = (
    #     UniqueConstraint('entity_id', name='uq_web_source_entity_id'),
    # )
    # If we expect source_url to be a primary way to avoid duplicates before an entity is made:
    # __table_args__ = (
    #     UniqueConstraint('source_url', name='uq_web_source_source_url'), # This would be too restrictive if multiple entities could point to same source URL for different reasons.
    # )
    # Usually, an entity is unique, and components are attached. If an entity can have only one WebSource, then (entity_id) unique.
    # If a source_url should only ever create one WebSourceComponent record globally (tied to one entity), then source_url unique.
    # The current model allows multiple WebSourceComponent for an entity and multiple entities pointing to the same URL (though services should prevent this).
    # For now, no additional table args. Service layer will manage uniqueness logic.
