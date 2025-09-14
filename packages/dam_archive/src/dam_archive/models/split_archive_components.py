from typing import Optional

from dam.models.core import BaseComponent as Component
from dam.models.core.component_mixins import UniqueComponentMixin
from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column


class SplitArchivePartInfoComponent(UniqueComponentMixin, Component):
    """
    Component to tag a file as a part of a split archive.
    """

    __tablename__ = "component_split_archive_part_info"

    part_num: Mapped[int] = mapped_column(Integer, nullable=False)
    master_entity_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("entities.id", name="fk_split_part_master_entity"),
        nullable=True,
        index=True,
        default=None,
    )


class SplitArchiveManifestComponent(UniqueComponentMixin, Component):
    """
    Component to hold the manifest of a complete split archive.
    This is typically attached to a master virtual entity.
    """

    __tablename__ = "component_split_archive_manifest"
