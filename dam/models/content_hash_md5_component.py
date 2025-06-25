from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base_component import BaseComponent


class ContentHashMD5Component(BaseComponent):
    """
    Stores MD5 content-based hashes for an entity.
    """

    __tablename__ = "component_content_hash_md5"

    hash_value: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # MD5 hashes are 32 chars

    __table_args__ = (UniqueConstraint("entity_id", "hash_value", name="uq_content_hash_md5_entity_hash"),)

    def __repr__(self):
        return (
            f"ContentHashMD5Component(id={self.id}, entity_id={self.entity_id}, hash_value='{self.hash_value[:10]}...')"
        )
