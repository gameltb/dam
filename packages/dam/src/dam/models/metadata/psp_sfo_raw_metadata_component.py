from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import BaseComponent


class PspSfoRawMetadataComponent(BaseComponent):
    """
    Stores raw metadata from a PSP ISO's PARAM.SFO file as a JSON object.
    """

    __tablename__ = "component_psp_sfo_raw_metadata"

    metadata_json: Mapped[dict] = mapped_column(JSON)

    def __repr__(self):
        return (
            f"PspSfoRawMetadataComponent(id={self.id}, entity_id={self.entity_id}, "
            f"metadata_json_keys='{list(self.metadata_json.keys())}')"
        )
