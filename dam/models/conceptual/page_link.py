from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.base_class import Base  # Inherit from Base for SQLAlchemy declarative mapping

if TYPE_CHECKING:
    from ..core.entity import Entity


class PageLink(Base):
    """
    Association object linking an "owner" Entity (which could be a comic book variant,
    or any other entity) to a "page image" Entity, with an order defined by page_number.
    This table facilitates a many-to-many relationship where an image can be a page
    in multiple owner entities, and an owner entity can have multiple ordered pages.
    """

    __tablename__ = "page_links"

    owner_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
        init=False,  # Explicitly make it not an __init__ arg
        comment="The ID of the Entity that owns this page list (e.g., an Entity with ComicBookVariantComponent).",
    )
    page_image_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
        init=False,  # Explicitly make it not an __init__ arg
        comment="The ID of the Entity representing the image for this page.",
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,  # This remains an __init__ arg
        comment="The order of this page within the owner's list (1-based).",
    )

    # Relationships to the Entity table for easy navigation from a PageLink instance
    owner: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[owner_entity_id],
        backref="owned_page_links",  # Entities will have 'owned_page_links' to access their PageLink entries as owner
    )
    page_image: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[page_image_entity_id],
        backref="image_page_links",  # Entities will have 'image_page_links' to access their PageLink entries as page_image
    )

    __table_args__ = (
        UniqueConstraint("owner_entity_id", "page_number", name="uq_owner_page_number"),
        # An image can appear only once for a given owner. If an image could be page 3 AND page 5
        # for the same owner, this constraint should be removed.
        # For now, assuming an image is used at most once per owner entity's page list.
        UniqueConstraint("owner_entity_id", "page_image_entity_id", name="uq_owner_page_image"),
    )

    def __repr__(self):
        return f"<PageLink owner_id={self.owner_entity_id} page_id={self.page_image_entity_id} page_num={self.page_number}>"
