from typing import Optional

from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


# @dataclass(kw_only=True) # kw_only=True is inherited from Base
class AudioPropertiesComponent(BaseComponent):
    """
    Component storing audio-specific metadata.
    """

    __tablename__ = "component_audio_properties"  # Adjusted table name
    # __mapper_args__ = {"polymorphic_identity": "audio_properties"} # Not needed

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    duration_seconds: Mapped[Optional[float]] = mapped_column(nullable=True, default=None)
    codec_name: Mapped[Optional[str]] = mapped_column(nullable=True, default=None)
    sample_rate_hz: Mapped[Optional[int]] = mapped_column(nullable=True, default=None)
    channels: Mapped[Optional[int]] = mapped_column(nullable=True, default=None)
    bit_rate_kbps: Mapped[Optional[int]] = mapped_column(nullable=True, default=None)  # e.g., 128, 192, 320

    # entity_id is inherited

    # Relationship to Entity is inherited from BaseComponent
    # entity: Mapped["Entity"] = relationship(back_populates="audio_properties_components")

    def __repr__(self):
        return (
            f"<AudioPropertiesComponent(id={self.id}, entity_id={self.entity_id}, "
            f"duration={self.duration_seconds}s, codec='{self.codec_name}', "
            f"sample_rate={self.sample_rate_hz}Hz, channels={self.channels})>"
        )
