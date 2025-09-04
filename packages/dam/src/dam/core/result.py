from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class HandlerResult(Generic[T]):
    """
    A container for the result of a single command handler.
    It holds either a successful value or an exception, mimicking the Result
    type in languages like Rust.
    """

    def __init__(self, value: Optional[T] = None, exception: Optional[Exception] = None):
        if exception is not None and value is not None:
            raise ValueError("HandlerResult cannot have both a value and an exception.")

        if exception is not None:
            self._is_ok = False
            self._exception: Optional[Exception] = exception
            self._value: Optional[T] = None
        else:
            self._is_ok = True
            self._value = value
            self._exception = None

    def is_ok(self) -> bool:
        """Returns True if the result is successful."""
        return self._is_ok

    def is_err(self) -> bool:
        """Returns True if the result is an error."""
        return not self._is_ok

    def unwrap(self) -> T:
        """
        Returns the contained value if the result is successful.
        Raises the contained exception if the result is an error.
        """
        if self.is_ok():
            return self._value
        raise self._exception

    @property
    def value(self) -> T:
        """
        Property to access the value. Alias for unwrap().
        """
        return self.unwrap()

    @property
    def exception(self) -> Optional[Exception]:
        """
        Property to access the exception if the result is an error.
        """
        return self._exception

    def __repr__(self) -> str:
        if self.is_ok():
            return f"Ok({self._value!r})"
        return f"Err({self._exception!r})"
