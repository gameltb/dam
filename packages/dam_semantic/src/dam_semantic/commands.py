from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from dam.core.commands import BaseCommand
from dam.system_events import BaseSystemEvent


@dataclass
class SemanticSearchCommand(BaseCommand[Optional[List[Tuple[Any, float, Any]]], BaseSystemEvent]):
    """A command to perform a semantic search for text."""

    query_text: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
