"""Defines the AudioPropertiesComponent model."""

from dam.models.core.base_component import BaseComponent
from sqlalchemy.orm import Mapped, mapped_column


# @dataclass(kw_only=True) # kw_only=True is inherited from Base
class AudioPropertiesComponent(BaseComponent):
    """Component storing audio-specific metadata."""

    __tablename__ = "component_audio_properties"  # Adjusted table name
    # __mapper_args__ = {"polymorphic_identity": "audio_properties"} # Not needed

    # id, entity_id are inherited from BaseComponent

    duration_seconds: Mapped[float | None] = mapped_column(nullable=True, default=None)
    codec_name: Mapped[str | None] = mapped_column(nullable=True, default=None)
    sample_rate_hz: Mapped[int | None] = mapped_column(nullable=True, default=None)
    channels: Mapped[int | None] = mapped_column(nullable=True, default=None)
    bit_rate_kbps: Mapped[int | None] = mapped_column(nullable=True, default=None)  # e.g., 128, 192, 320

