# This file makes the 'models' directory a Python package.

# Ensure all model modules are imported so they register with Base.metadata
import dam.core.components_markers  # For marker components like NeedsMetadataExtractionComponent

from .audio_properties_component import AudioPropertiesComponent
from .base_class import Base  # Import Base from its new location
from .base_component import BaseComponent
from .content_hash_md5_component import ContentHashMD5Component
from .content_hash_sha256_component import ContentHashSHA256Component
from .entity import Entity
from .file_location_component import FileLocationComponent
from .file_properties_component import FilePropertiesComponent
from .frame_properties_component import FramePropertiesComponent
from .image_dimensions_component import ImageDimensionsComponent
from .image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from .image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from .image_perceptual_hash_phash_component import ImagePerceptualPHashComponent
from .original_source_info_component import OriginalSourceInfoComponent
from .web_source_component import WebSourceComponent
from .website_profile_component import WebsiteProfileComponent # Import new component


# Optionally, define an __all__ for explicit public API of the models package
__all__ = [
    "Base",
    "Entity",
    "BaseComponent",
    # Core Components
    "ContentHashMD5Component",
    "ContentHashSHA256Component",
    "FileLocationComponent",
    "FilePropertiesComponent",
    "OriginalSourceInfoComponent",
    "WebSourceComponent",
    "WebsiteProfileComponent", # Added new component
    # Media-specific property components
    "AudioPropertiesComponent",
    "FramePropertiesComponent",
    "ImageDimensionsComponent",
    # Perceptual hash components
    "ImagePerceptualAHashComponent",
    "ImagePerceptualDHashComponent",
    "ImagePerceptualPHashComponent",
    # Marker components are not typically part of __all__ unless directly used by external modules
    # but their import via dam.core.components_markers is crucial.
]
