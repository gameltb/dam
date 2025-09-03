from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


class ResourceNotFoundError(Exception):
    """Exception raised when a requested resource is not found."""

    def __init__(self, resource_type: Type):
        super().__init__(f"Resource of type {resource_type.__name__} not found.")


class ResourceManager:
    """
    A simple container for managing and providing access to global or shared resources.
    Resources are typically singleton instances of function modules (or wrappers around
    function modules) or other utility objects required by systems.

    The `ResourceManager` allows systems to declare dependencies on resources
    via type hints, which are then injected by the `WorldScheduler`.
    """

    def __init__(self):
        """Initializes an empty ResourceManager."""
        self._resources: Dict[Type, Any] = {}

    def add_resource(self, instance: Any, resource_type: Type[T] = None) -> None:
        """
        Adds a resource instance to the manager.

        The resource is typically registered using its own class type as `resource_type`.
        If `resource_type` is not provided, it's inferred from the `instance`'s direct type.
        If a resource of the same type already exists, it will be replaced, and a
        warning will be printed.

        Args:
            instance: The resource instance to add (e.g., an instance of `FileOperationsResource`).
            resource_type: The type (class) to register this instance against. Systems will
                           request resources using this type. Defaults to `type(instance)`.
        """
        if resource_type is None:
            resource_type = type(instance)

        if resource_type in self._resources:
            # Depending on policy, could raise error, log a warning, or allow replacement.
            # For now, let's log a warning and replace.
            # Consider making this behavior configurable if needed.
            print(f"Warning: Replacing existing resource for type {resource_type.__name__}")

        self._resources[resource_type] = instance

    def get_resource(self, resource_type: Type[T]) -> T:
        """
        Retrieves a resource instance by its registered type.

        If a direct match for `resource_type` is not found, this method will also
        attempt to find a registered resource that is a subclass of the requested `resource_type`.

        Args:
            resource_type: The type of the resource to retrieve.

        Returns:
            The resource instance.

        Raises:
            ResourceNotFoundError: If no resource matching the `resource_type` (or its subclass)
                                   is found.
        """
        instance = self._resources.get(resource_type)
        if instance is None:
            # Try to find a subclass instance if direct type match fails
            for res_type, res_instance in self._resources.items():
                if issubclass(res_type, resource_type):  # Check if registered type is a subclass of requested type
                    return res_instance  # type: ignore
            raise ResourceNotFoundError(resource_type)
        return instance  # type: ignore

    def has_resource(self, resource_type: Type) -> bool:
        """
        Checks if a resource of the given type (or a subclass of it) is registered.
        """
        if resource_type in self._resources:
            return True
        for res_type in self._resources:
            if issubclass(res_type, resource_type):
                return True
        return False


