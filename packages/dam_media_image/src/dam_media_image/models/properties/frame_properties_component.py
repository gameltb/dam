"""Defines the FramePropertiesComponent for storing metadata about animated images."""

from dam.models.core.base_component import BaseComponent
from sqlalchemy.orm import Mapped, mapped_column


class FramePropertiesComponent(BaseComponent):
    """
    Component storing metadata for sequences of frames, such as animated GIFs or video visual tracks.

    This includes frame count, duration of the sequence, and nominal frame rate.
    Dimensions (width/height) for these visual assets are stored in ImageDimensionsComponent.
    """

    __tablename__ = "component_frame_properties"

    frame_count: Mapped[int | None] = mapped_column(nullable=True, default=None)
    nominal_frame_rate: Mapped[float | None] = mapped_column(nullable=True, default=None)  # Frames per second
    animation_duration_seconds: Mapped[float | None] = mapped_column(nullable=True, default=None)

