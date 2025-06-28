from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text

from dam.models.core.base_component import BaseComponent
from dam.models.conceptual.base_conceptual_info_component import BaseConceptualInfoComponent
from sqlalchemy import ForeignKey # Added for ForeignKey


class EvaluationRunComponent(BaseConceptualInfoComponent): # Removed BaseComponent
    """
    Component defining an evaluation run.
    This is a conceptual asset, representing the concept of a specific
    evaluation setup (e.g., evaluating various profiles on a set of assets).
    """
    __tablename__ = "evaluation_runs"

    # This 'id' is the primary key of this table AND a foreign key to entities.id
    id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)

    run_name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    # concept_name from BaseConceptualInfoComponent will be the run_name

    # concept_description from BaseConceptualInfoComponent can be used for a general description
    # For more detailed information, like the list of source assets or profiles used,
    # it's better to store them in related EvaluationResultComponents or a serialized field if necessary.
    # For now, we'll rely on querying EvaluationResultComponents linked to this run.

    # Fields from BaseConceptualInfoComponent:
    # concept_name: Mapped[str] (will be run_name)
    # concept_description: Mapped[str | None] (can be used for general description)

    __mapper_args__ = {
        "polymorphic_identity": "evaluation_run",
        # For joined table, inherit_condition is often not needed if PK/FK is clear.
    }

    def __repr__(self) -> str:
        return (
            f"<EvaluationRunComponent(id={self.id}, entity_id={self.entity_id}, "
            f"run_name='{self.run_name}')>"
        )
