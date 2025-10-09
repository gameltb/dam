"""Defines the TranscodedVariantComponent for linking transcoded assets to their originals."""

from dam.models.core.base_component import BaseComponent
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class TranscodedVariantComponent(BaseComponent):
    """
    Component linking a transcoded asset to its original asset and the transcoding profile used.

    It can also store quality and size metrics. This component is attached to the entity
    representing the transcoded file.
    """

    __tablename__ = "component_transcoded_variant"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True, init=False)

    original_asset_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id"), nullable=False, index=True
    )
    transcode_profile_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id"), nullable=False, index=True
    )

    transcoded_file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_metric_vmaf: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_metric_ssim: Mapped[float | None] = mapped_column(Float, nullable=True)
    custom_metrics_json: Mapped[str | None] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return (
            f"<TranscodedVariantComponent(id={self.id}, entity_id={self.entity_id}, "
            f"original_entity_id={self.original_asset_entity_id}, "
            f"profile_entity_id={self.transcode_profile_entity_id}, "
            f"size_bytes={self.transcoded_file_size_bytes})>"
        )
