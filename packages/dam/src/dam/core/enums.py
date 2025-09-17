# pyright: basic
from enum import Enum, auto


class ExecutionStrategy(Enum):
    """
    Defines the execution strategy for a set of systems.
    """

    SERIAL = auto()
    PARALLEL = auto()


class SystemType(Enum):
    """
    Defines the type of a system, which determines how it is triggered.
    """

    VANILLA = auto()
    STAGE = auto()
    COMMAND = auto()
    EVENT = auto()
