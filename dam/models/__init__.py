# This file makes the 'models' directory a Python package.

from .audio_properties_component import AudioPropertiesComponent
from .base_class import Base  # Import Base from its new location
from .base_component import BaseComponent
from .content_hash_md5_component import ContentHashMD5Component
from .content_hash_sha256_component import ContentHashSHA256Component

# Import models here to ensure they are registered with SQLAlchemy's metadata
# and to make them easily accessible from dam.models
from .entity import Entity
from .file_location_component import FileLocationComponent
from .file_properties_component import FilePropertiesComponent
from .frame_properties_component import FramePropertiesComponent
from .image_dimensions_component import ImageDimensionsComponent  # Added
from .image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from .image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from .image_perceptual_hash_phash_component import ImagePerceptualPHashComponent

# Optionally, define an __all__ for explicit public API of the models package
__all__ = [
    "Base",
    "Entity",
    "BaseComponent",
    "ContentHashMD5Component",
    "ContentHashSHA256Component",
    "ImagePerceptualAHashComponent",
    "ImagePerceptualDHashComponent",
    "ImagePerceptualPHashComponent",
    "FileLocationComponent",
    "FilePropertiesComponent",
    # "VideoPropertiesComponent", # Removed
    "AudioPropertiesComponent",
    "FramePropertiesComponent",
    "ImageDimensionsComponent",  # Added
]
