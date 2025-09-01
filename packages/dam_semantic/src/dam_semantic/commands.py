from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from dam.core.commands import BaseCommand


@dataclass
class SemanticSearchCommand(BaseCommand[Optional[List[Tuple[Any, float, Any]]]]):
    """A command to perform a semantic search for text."""

    query_text: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
