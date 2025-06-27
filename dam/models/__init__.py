# This file makes the 'models' directory a Python package.

# Ensure all model modules are imported so they register with Base.metadata
import dam.core.components_markers  # For marker components

# Core model elements
from .core.base_class import Base
from .core.base_component import BaseComponent
from .core.entity import Entity
from .core.file_location_component import FileLocationComponent
from .core.types import BinaryHashValue, HexStringHashValue, PerceptualHashValue, StorageType

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

__all__ = [
    # Core
    "Base",
    "Entity",
    "BaseComponent",
    "FileLocationComponent",
    "BinaryHashValue",
    "HexStringHashValue",
    "PerceptualHashValue",
    "StorageType",
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
    # Marker components are usually imported via dam.core.components_markers where needed
]
