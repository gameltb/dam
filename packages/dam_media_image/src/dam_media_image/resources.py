"""Defines the resource for the image hashing service."""
from .functions import image_hashing_functions as image_hashing_service


class ImageHashingServiceResource:
    """A resource that provides access to image hashing functions."""

    def __init__(self) -> None:
        """Initialize the resource by binding the hashing functions."""
        self.generate_perceptual_hashes = image_hashing_service.generate_perceptual_hashes
        self.generate_perceptual_hashes_async = image_hashing_service.generate_perceptual_hashes_async
