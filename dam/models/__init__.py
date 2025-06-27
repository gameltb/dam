# This file makes the 'models' directory a Python package.

# Ensure all model modules are imported so they register with Base.metadata
from ..core.components_markers import *  # noqa: F403

# Core model elements
from .core.base_class import Base
from .core.base_component import BaseComponent
from .core.entity import Entity
from .core.file_location_component import FileLocationComponent

# Hash components
from .hashes.content_hash_md5_component import ContentHashMD5Component
from .hashes.content_hash_sha256_component import ContentHashSHA256Component
from .hashes.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from .hashes.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from .hashes.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent

# Property components
from .properties.audio_properties_component import AudioPropertiesComponent
from .properties.file_properties_component import FilePropertiesComponent
from .properties.frame_properties_component import FramePropertiesComponent
from .properties.image_dimensions_component import ImageDimensionsComponent

# Source info components
from .source_info.original_source_info_component import OriginalSourceInfoComponent
from .source_info.web_source_component import WebSourceComponent
from .source_info.website_profile_component import WebsiteProfileComponent

# Conceptual components
# Import the concrete components that are intended for direct use/instantiation
from .conceptual.comic_book_concept_component import ComicBookConceptComponent
from .conceptual.comic_book_variant_component import ComicBookVariantComponent
from .conceptual.page_link import PageLink # Added PageLink
# Base abstract conceptual components like BaseConceptualInfoComponent and BaseVariantInfoComponent
# are typically not re-exported here unless specifically needed for widespread type hinting.
# Their subclasses (the concrete components above) are what services and systems will primarily work with.

__all__ = [
    # Core
    "Base",
    "Entity",
    "BaseComponent",
    "FileLocationComponent",
    # Hashes
    "ContentHashMD5Component",
    "ContentHashSHA256Component",
    "ImagePerceptualAHashComponent",
    "ImagePerceptualDHashComponent",
    "ImagePerceptualPHashComponent",
    # Properties
    "AudioPropertiesComponent",
    "FilePropertiesComponent",
    "FramePropertiesComponent",
    "ImageDimensionsComponent",
    # Source Info
    "OriginalSourceInfoComponent",
    "WebSourceComponent",
    "WebsiteProfileComponent",
    # Conceptual (Concrete Components and Association Objects)
    "ComicBookConceptComponent",
    "ComicBookVariantComponent",
    "PageLink", # Added PageLink
    # Marker components are usually imported via dam.core.components_markers where needed
]
