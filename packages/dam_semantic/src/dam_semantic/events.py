import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from dam.core.events import BaseEvent


@dataclass
class SemanticSearchQuery(BaseEvent):
    query_text: str
    world_name: str
    request_id: str
    top_n: int = 10
    model_name: Optional[str] = None  # Uses service default if None
    # For SemanticSearchQuery, the result_future will yield List[Tuple[Entity, float, TextEmbeddingComponent]]
    # We use 'Any' here to avoid circular dependencies with model imports at the event definition level.
    result_future: Optional[asyncio.Future[List[Tuple[Any, float, Any]]]] = field(default=None, init=False, repr=False)
