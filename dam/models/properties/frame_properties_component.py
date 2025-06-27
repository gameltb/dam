from typing import Optional

from sqlalchemy.orm import Mapped, mapped_column

from .. import BaseComponent


# @dataclass(kw_only=True) # kw_only=True is inherited from Base
class FramePropertiesComponent(BaseComponent):
    """
    Component storing metadata for sequences of frames, such as animated GIFs or video visual tracks.
    This includes frame count, duration of the sequence, and nominal frame rate.
    Dimensions (width/height) for these visual assets are stored in ImageDimensionsComponent.
    """

    __tablename__ = "component_frame_properties"  # Adjusted table name
    # __mapper_args__ = {"polymorphic_identity": "frame_properties"}
    # Not needed if not using Single Table Inheritance with BaseComponent as parent table

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    frame_count: Mapped[Optional[int]] = mapped_column(nullable=True, default=None)
    # Frame rate might be tricky as it can vary per frame in formats like GIF.
    # Storing an average or typical frame rate might be an option, or duration of the animation.
    # For simplicity, let's assume a single representative frame rate or overall duration for now.
    nominal_frame_rate: Mapped[Optional[float]] = mapped_column(nullable=True, default=None)  # Frames per second
    animation_duration_seconds: Mapped[Optional[float]] = mapped_column(nullable=True, default=None)
    # image_format: Mapped[Optional[str]] = mapped_column(nullable=True) # e.g., "GIF", "APNG", "AVIF sequence"
    # This might be better in FilePropertiesComponent or a dedicated format component

    # Dimensions (width, height) are often part of image components or file properties.
    # If animated images always have consistent dimensions, it can be stored there.
    # If dimensions can vary per frame (less common for standard formats like GIF),
    # then width/height might be needed here or in a per-frame sub-component.
    # For now, assuming consistent dimensions stored elsewhere (e.g. ImageDimensionComponent if applicable)

    # entity_id is inherited from BaseComponent

    def __repr__(self):
        return (
            f"<FramePropertiesComponent(id={self.id}, entity_id={self.entity_id}, "
            f"frames={self.frame_count}, duration_sec={self.animation_duration_seconds}, "
            f"fps={self.nominal_frame_rate})>"
        )
