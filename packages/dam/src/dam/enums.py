# pyright: basic
from enum import Enum, auto


class ExecutionStrategy(Enum):
    """Defines the execution strategy for a set of systems."""

    SERIAL = auto()
    PARALLEL = auto()
