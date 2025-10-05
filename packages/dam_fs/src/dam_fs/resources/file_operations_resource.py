from ..functions import file_operations as ops


class FileOperationsResource:
    """
    A resource that provides access to file system operations.

    This class acts as a wrapper around the functions defined in the
    `dam_fs.functions.file_operations` module. It makes these operations
    available for dependency injection into systems via the `ResourceManager`.

    Systems can request this resource by type-hinting a parameter:
    `file_ops: Annotated[FileOperationsResource, "Resource"]`
    """

    def __init__(self) -> None:
        """
        Initializes the FileOperationsResource by binding methods to the
        functions from the `dam_fs.functions.file_operations` module.
        """
        # Make functions available as methods of this resource instance
        self.get_file_properties = ops.get_file_properties
        self.get_mime_type = ops.get_mime_type
        self.read_file_async = ops.read_file_async
        self.get_file_properties_async = ops.get_file_properties_async
        self.get_mime_type_async = ops.get_mime_type_async
        # Add other relevant functions from file_operations as needed by systems.

    # Note: If dam_fs.functions.file_operations was a class, this resource could simply
    # be an instance of that class, or this class could inherit from it.
    # Since it's a module of functions, this wrapper approach is used to make
    # them available as an injectable resource object.
