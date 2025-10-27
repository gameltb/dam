"""Defines the EvaluationRunComponent for representing a transcoding evaluation run."""

from dam.models.conceptual.base_conceptual_info_component import BaseConceptualInfoComponent
from sqlalchemy import (
    ForeignKey,  # Added for ForeignKey
    String,
)
from sqlalchemy.orm import Mapped, mapped_column


class EvaluationRunComponent(BaseConceptualInfoComponent):  # Removed BaseComponent
    """
    Component defining an evaluation run.

    This is a conceptual asset, representing the concept of a specific
    evaluation setup (e.g., evaluating various profiles on a set of assets).
    """

    __tablename__ = "component_evaluation_run"  # Renamed

    # This 'id' is the primary key of this table AND a foreign key to entities.id
    id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)

    run_name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
