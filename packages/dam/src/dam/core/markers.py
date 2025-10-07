"""Marker classes for dependency injection."""
# pyright: basic


class ResourceMarker:
    """A marker class for injecting resources."""

    pass


class EventMarker:
    """A marker class for injecting events."""

    pass


class CommandMarker:
    """A marker class for injecting commands."""

    pass


class MarkedEntityList:
    """A marker class for injecting a list of entities with a specific component."""

    pass
