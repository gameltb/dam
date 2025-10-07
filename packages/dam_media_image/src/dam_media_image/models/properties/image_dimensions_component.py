
from dam.models.core.base_component import BaseComponent
from sqlalchemy import Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


# kw_only=True is inherited from Base via BaseComponent
class ImageDimensionsComponent(BaseComponent):
    """
    Stores width and height for a visual entity (e.g., static image,
    video frame dimensions, GIF dimensions).
    An entity typically has one primary set of dimensions.
    """

    __tablename__ = "component_image_dimensions"

    width_pixels: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    height_pixels: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    __table_args__ = (
        # An entity should generally have only one primary dimensions component.
        UniqueConstraint("entity_id", name="uq_image_dimensions_entity_id"),
    )

    def __repr__(self) -> str:
        return f"<ImageDimensionsComponent(id={self.id}, entity_id={self.entity_id}, width={self.width_pixels}, height={self.height_pixels})>"
