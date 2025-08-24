import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from dam.core.commands import BaseCommand


@dataclass
class SemanticSearchCommand(BaseCommand):
    """A command to perform a semantic search for text."""

    query_text: str
    world_name: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
    result_future: Optional[asyncio.Future[List[Tuple[Any, float, Any]]]] = field(default=None, init=False, repr=False)
