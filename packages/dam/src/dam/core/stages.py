from enum import Enum, auto


class SystemStage(Enum):
    """
    Defines distinct stages in the application lifecycle where systems can be executed.
    Systems are registered to run at specific stages using the `@system(stage=SystemStage.SOME_STAGE)`
    decorator. The `WorldScheduler` then executes systems belonging to a requested stage,
    typically in the order these enum members are defined.

    The choice and order of stages depend on the application's workflow.
    """

    # Example Stages - these can be refined based on application needs

    PRE_PROCESSING = auto()  # Systems that run before main logic (e.g., input validation, setup)
    ASSET_INGESTION = auto()  # Stage where asset_service primarily operates ( synchronous part)
    # (e.g. creating entity, core components, adding marker components)

    METADATA_EXTRACTION = auto()  # Systems for extracting metadata (e.g., Hachoir, EXIF)
    # Typically runs after ASSET_INGESTION and processes marked entities.

    CONTENT_ANALYSIS = auto()  # Systems for deeper analysis (e.g., image similarity, AI tagging)
    # May run after METADATA_EXTRACTION.

    POST_PROCESSING = auto()  # Systems that run after main logic (e.g., cleanup, notifications)

    # CLI specific stages if needed
    CLI_COMMAND_START = auto()
    CLI_COMMAND_END = auto()

    # General purpose update stage if the application had a main loop
    # UPDATE = auto()

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            # Get all members in definition order
            members = list(self.__class__)
            return members.index(self) < members.index(other)
        return NotImplemented
