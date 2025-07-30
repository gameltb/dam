class SireError(Exception):
    """Base exception for all errors in the sire library."""

    pass


class ModelNotFoundError(SireError):
    """Raised when a model is not found in the registry."""

    pass


class InsufficientMemoryError(SireError):
    """Raised when there is not enough memory to load a model."""

    pass


class InferenceError(SireError):
    """Raised when an error occurs during model inference."""

    pass


class ModelNotLoadedError(SireError):
    """Raised when trying to perform an action on a model that is not loaded."""

    pass
