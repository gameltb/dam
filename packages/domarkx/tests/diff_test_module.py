"""A test module for diffing methods in classes."""


class BaseClass:
    """A base class for testing method diffing."""

    def greeting(self) -> str:
        """Return a simple greeting."""
        return "Hello from BaseClass"

    def farewell(self) -> str:
        """Return a farewell message."""
        return "Goodbye from BaseClass"


class SubClass(BaseClass):
    """A subclass that overrides the greeting method."""

    def greeting(self) -> str:
        """Return a more enthusiastic greeting."""
        return "Hello from SubClass!"


class SubClassWithNoOverride(BaseClass):
    """A subclass that does not override any methods."""

    def new_method(self) -> str:
        """Return a new method's string."""
        return "This is a new method."


class SubClassWithIdenticalOverride(BaseClass):
    """A subclass that overrides a method with an identical implementation."""

    def greeting(self) -> str:
        """Return a simple greeting."""
        return "Hello from BaseClass"


class GrandChildClass(SubClass):
    """A class that inherits from SubClass to test deeper inheritance."""

    def farewell(self) -> str:
        """Return a more complex farewell."""
        return "Farewell, and may the force be with you."
