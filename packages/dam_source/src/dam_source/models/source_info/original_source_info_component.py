"""Defines the OriginalSourceInfoComponent model."""

from dam.models.core import BaseComponent
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from . import source_types


class OriginalSourceInfoComponent(BaseComponent):
    """
    Classify the original source type of an entity's content.

    This component acts as a marker or tag, indicating the nature of the origin.
    Detailed information about the source, like filename, path, or URL,
    is stored in other components such as FilePropertiesComponent,
    FileLocationComponent, or WebSourceComponent.

    An entity typically has one OriginalSourceInfoComponent indicating its
    primary mode of ingestion or creation.
    """

    __tablename__ = "component_original_source_info"

    # id, entity_id are inherited from BaseComponent

    # Fields like original_filename and original_path have been removed.
    # This information is now expected to be found in:
    # - FilePropertiesComponent.original_filename
    # - FileLocationComponent.path (for DAM managed paths or external reference paths)
    # - WebSourceComponent.url (for web origins)

    source_type: Mapped[str] = mapped_column(
        String(),
        nullable=False,
        index=True,
        comment=(
            "Type classifying the source. See dam_source.models.source_info.source_types for defined constants "
            f"(e.g., '{source_types.SOURCE_TYPE_LOCAL_FILE}', '{source_types.SOURCE_TYPE_REFERENCED_FILE}', "
            f"'{source_types.SOURCE_TYPE_WEB_SOURCE}', '{source_types.SOURCE_TYPE_PRIMARY_FILE}')."
        ),
    )
