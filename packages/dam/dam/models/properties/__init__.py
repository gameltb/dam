# This file makes the 'properties' directory a Python package.

from .audio_properties_component import AudioPropertiesComponent  # Assuming this should be exported
from .file_properties_component import FilePropertiesComponent
from .frame_properties_component import FramePropertiesComponent  # Assuming this should be exported
from .image_dimensions_component import ImageDimensionsComponent  # Assuming this should be exported

__all__ = [
    "FilePropertiesComponent",
    "ImageDimensionsComponent",
    "FramePropertiesComponent",
    "AudioPropertiesComponent",
]
