from typing import Optional

from sqlalchemy import BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from dam.models.core import UniqueComponent


class ContentLengthComponent(UniqueComponent):
    """
    Stores the size of the asset's content in bytes.
    """

    __tablename__ = "component_content_length"

    # entity_id is inherited from UniqueComponent

    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, default=None)

    def __repr__(self) -> str:
        return (
            f"ContentLengthComponent(entity_id={self.entity_id}, file_size_bytes={self.file_size_bytes})"
        )
