from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core.base_component import BaseComponent


class EvaluationResultComponent(BaseComponent):
    """
    Component storing the results of a specific transcoding operation
    performed as part of an evaluation run.
    This component is attached to the entity representing the transcoded file
    that was generated during the evaluation.
    """

    __tablename__ = "component_evaluation_result"  # Renamed

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True, init=False)
    # entity_id is inherited from BaseComponent, linking this to the transcoded asset's entity.

    evaluation_run_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id"), nullable=False, index=True
    )
    # This is the entity ID of the EvaluationRunComponent conceptual asset.

    original_asset_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id"), nullable=False, index=True
    )
    # The original asset that was transcoded.

    transcode_profile_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id"), nullable=False, index=True
    )
    # The transcode profile (conceptual asset entity ID) used for this specific result.

    # Transcoded file specific info (can be redundant with TranscodedVariantComponent, but useful here for direct reporting)
    transcoded_asset_entity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
        unique=True,
        # Unique because one evaluation result per transcoded asset from an eval run
    )
    # This is the entity_id of the asset this component is attached to.

    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Quality Metrics (examples, can be expanded)
    vmaf_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ssim_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    psnr_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Field for other/custom metrics, stored as JSON string
    custom_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Notes or comments specific to this evaluation result
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "evaluation_result",
        "inherit_condition": id == BaseComponent.id,  # type: ignore
    }

    def __repr__(self) -> str:
        return (
            f"<EvaluationResultComponent(id={self.id}, entity_id={self.entity_id}, "
            f"run_id={self.evaluation_run_entity_id}, original_id={self.original_asset_entity_id}, "
            f"profile_id={self.transcode_profile_entity_id}, size={self.file_size_bytes}, vmaf={self.vmaf_score})>"
        )
