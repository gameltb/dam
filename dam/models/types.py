from typing import Annotated
from datetime import datetime

from sqlalchemy import text, DateTime
from sqlalchemy.orm import mapped_column

# Timestamp for creation, defaults to the current time on the database side
TimestampCreated = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        # init=False removed as per SADeprecationWarning
    ),
]

# Timestamp for updates, defaults to the current time and updates on every modification
TimestampUpdated = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        # init=False removed as per SADeprecationWarning
    ),
]

# Generic Primary Key type
# init=False should be handled by field(init=False) on the attribute itself.
PkId = Annotated[int, mapped_column(primary_key=True, autoincrement=True)]
