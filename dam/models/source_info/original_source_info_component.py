from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.base_component import BaseComponent  # Corrected import
from . import source_types


class OriginalSourceInfoComponent(BaseComponent):
    """
    Classifies the original source type of an entity's content.
    This component acts as a marker or tag, indicating the nature of the origin.
    Detailed information about the source, like filename, path, or URL,
    is stored in other components such as FilePropertiesComponent,
    FileLocationComponent, or WebSourceComponent.

    An entity typically has one OriginalSourceInfoComponent indicating its
    primary mode of ingestion or creation.
    """

    __tablename__ = "component_original_source_info"

    # id, entity_id, created_at, updated_at are inherited from BaseComponent

    # Fields like original_filename and original_path have been removed.
    # This information is now expected to be found in:
    # - FilePropertiesComponent.original_filename
    # - FileLocationComponent.path (for DAM managed paths or external reference paths)
    # - WebSourceComponent.url (for web origins)

    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment=(
            "Type classifying the source. See dam.models.source_info.source_types for defined constants "
            f"(e.g., '{source_types.SOURCE_TYPE_LOCAL_FILE}', '{source_types.SOURCE_TYPE_REFERENCED_FILE}', "
            f"'{source_types.SOURCE_TYPE_WEB_SOURCE}', '{source_types.SOURCE_TYPE_PRIMARY_FILE}')."
        ),
    )

    def __repr__(self):
        return (
            f"OriginalSourceInfoComponent(id={self.id}, entity_id={self.entity_id}, source_type='{self.source_type}')"
        )
