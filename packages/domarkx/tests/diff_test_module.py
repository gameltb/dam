class BaseClass:
    """A base class for testing method diffing."""

    def greeting(self) -> str:
        """Returns a simple greeting."""
        return "Hello from BaseClass"

    def farewell(self) -> str:
        """Returns a farewell message."""
        return "Goodbye from BaseClass"


class SubClass(BaseClass):
    """A subclass that overrides the greeting method."""

    def greeting(self) -> str:
        """Returns a more enthusiastic greeting."""
        return "Hello from SubClass!"


class SubClassWithNoOverride(BaseClass):
    """A subclass that does not override any methods."""

    def new_method(self) -> str:
        return "This is a new method."


class SubClassWithIdenticalOverride(BaseClass):
    """A subclass that overrides a method with an identical implementation."""

    def greeting(self) -> str:
        """Returns a simple greeting."""
        return "Hello from BaseClass"


class GrandChildClass(SubClass):
    """A class that inherits from SubClass to test deeper inheritance."""

    def farewell(self) -> str:
        """A more complex farewell."""
        message = "Farewell, and may the force be with you."
        return message
