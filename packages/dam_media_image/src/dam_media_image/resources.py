from .services import image_hashing_service

class ImageHashingServiceResource:
    def __init__(self):
        self.generate_perceptual_hashes = image_hashing_service.generate_perceptual_hashes
        self.generate_perceptual_hashes_async = image_hashing_service.generate_perceptual_hashes_async
