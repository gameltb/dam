# This file makes the 'models' directory a Python package.

from .base_class import Base  # Import Base from its new location
from .base_component import BaseComponent
from .content_hash_component import ContentHashComponent

# Import models here to ensure they are registered with SQLAlchemy's metadata
# and to make them easily accessible from dam.models
from .entity import Entity
from .file_location_component import FileLocationComponent
from .file_properties_component import FilePropertiesComponent
from .image_perceptual_hash_component import ImagePerceptualHashComponent

# Optionally, define an __all__ for explicit public API of the models package
__all__ = [
    "Base",
    "Entity",
    "BaseComponent",
    "ContentHashComponent",
    "ImagePerceptualHashComponent",
    "FileLocationComponent",
    "FilePropertiesComponent",
]
