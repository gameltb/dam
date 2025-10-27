"""Links a virtual ISO entity back to the original CSO file entity from which it was derived."""

from dam.models.core.base_component import BaseComponent
from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


class CsoParentIsoComponent(BaseComponent):
    """Links a virtual ISO entity back to the original CSO file entity from which it was derived."""

    __tablename__ = "component_cso_parent_iso"

    cso_entity_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("entities.id"),
        index=True,
        unique=True,
        nullable=False,
    )

