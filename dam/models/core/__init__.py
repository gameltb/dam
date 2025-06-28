# This file makes the 'core' directory a Python package.

# Example:
from .entity import Entity
from .base_component import BaseComponent # Assuming BaseComponent should also be exported
from .file_location_component import FileLocationComponent # Assuming FileLocationComponent should also be exported

__all__ = ["Entity", "BaseComponent", "FileLocationComponent"]
