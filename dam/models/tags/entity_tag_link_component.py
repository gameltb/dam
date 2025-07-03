from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.base_component import BaseComponent  # This links an entity to a tag concept

if TYPE_CHECKING:
    from ..core.entity import Entity


class EntityTagLinkComponent(BaseComponent):
    """
    Links an Entity (the one this component is attached to) to a TagConceptEntity
    (an Entity that has a TagConceptComponent, representing the tag definition).
    This component effectively applies a defined tag to an entity, optionally with a value.
    """

    __tablename__ = "component_entity_tag_link"

    # entity_id is inherited from BaseComponent - this is the ID of the entity being tagged.
    # id, created_at, updated_at are also inherited.

    tag_concept_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        init=False,  # Make this init=False as it's set via the 'tag_concept' relationship
        comment="The ID of the Entity that has the TagConceptComponent (i.e., the tag definition).",
    )

    tag_value: Mapped[str | None] = mapped_column(
        String(1024),  # Max length for a tag value
        nullable=True,
        comment="Optional value for this tag application, used if the TagConcept allows values (e.g., for a 'Rating' tag, value could be '5').",
    )

    # Relationship to the TagConcept's Entity for easier navigation
    tag_concept: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[tag_concept_entity_id],
        backref="applied_as_tag_links",  # Keep simple backref name
        passive_deletes=True,  # Instructs SA to not attempt to NULL this FK on delete of parent
    )

    __table_args__ = (
        # An entity can't have the exact same tag_concept applied with the exact same value multiple times.
        # If tag_value is NULL, then a specific tag_concept can only be applied once to an entity.
        # Database handling of NULLs in unique constraints can vary.
        # For instance, PostgreSQL treats NULLs as distinct, allowing multiple NULLs.
        # SQLite typically allows multiple NULLs in unique constraints as well.
        # If stricter "apply once if no value" is needed, service layer logic might be required
        # or a more complex constraint / generated column.
        UniqueConstraint("entity_id", "tag_concept_entity_id", "tag_value", name="uq_entity_tag_application"),
    )

    def __repr__(self):
        return (
            f"<EntityTagLinkComponent id={self.id} entity_id={self.entity_id} "
            f"tag_concept_id={self.tag_concept_entity_id} value='{self.tag_value}'>"
        )
