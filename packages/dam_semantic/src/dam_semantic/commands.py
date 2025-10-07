from dataclasses import dataclass
from typing import Any, Optional

from dam.commands.core import BaseCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class SemanticSearchCommand(BaseCommand[Optional[list[tuple[Any, float, Any]]], BaseSystemEvent]):
    """A command to perform a semantic search for text."""

    query_text: str
    request_id: str
    top_n: int = 10
    model_name: str | None = None  # Uses service default if None
