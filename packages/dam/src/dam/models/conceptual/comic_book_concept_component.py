from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.component_mixins import UniqueComponentMixin
from .base_conceptual_info_component import BaseConceptualInfoComponent


class ComicBookConceptComponent(UniqueComponentMixin, BaseConceptualInfoComponent):
    """
    Represents the concept of a specific comic book issue or work.
    This component is attached to an Entity that acts as the central identifier
    for this comic book concept, to which different file variants can be linked.
    """

    __tablename__ = "component_comic_book_concept"

    # id, entity_id are inherited from BaseComponent
    # via BaseConceptualInfoComponent.

    comic_title: Mapped[str] = mapped_column(
        String(),
        nullable=False,
        index=True,
        comment="The main title of the comic book (e.g., 'The Amazing Spider-Man', 'Action Comics').",
    )

    series_title: Mapped[str | None] = mapped_column(
        String(),
        nullable=True,
        index=True,
        comment="The title of the series this comic belongs to, if applicable (e.g., 'The Amazing Spider-Man', 'Action Comics'). Often same as comic_title for ongoing series.",
    )

    issue_number: Mapped[str | None] = mapped_column(
        String(),
        nullable=True,
        index=True,
        comment="The issue number, if applicable (e.g., '#1', '100', 'Annual #3').",
    )

    publication_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="The year the comic book issue was published.",
    )

    # Could add other relevant comic-specific conceptual fields:
    # publisher: Mapped[str | None]
    # volume_number: Mapped[int | None]
    # story_arc_title: Mapped[str | None]
    # summary: Mapped[str | None] # Kept as Text in SQLAlchemy

    def __repr__(self) -> str:
        return (
            f"ComicBookConceptComponent(id={self.id}, entity_id={self.entity_id}, "
            f"comic_title='{self.comic_title}', issue='{self.issue_number}', year={self.publication_year})"
        )
