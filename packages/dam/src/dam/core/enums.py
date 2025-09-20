# pyright: basic
from enum import Enum, auto


class SystemType(Enum):
    """
    Defines the type of a system, which determines how it is triggered.
    """

    VANILLA = auto()
    STAGE = auto()
    COMMAND = auto()
    EVENT = auto()
