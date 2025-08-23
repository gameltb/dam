import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.tags import (
    ModelGeneratedTagLinkComponent,
)
from dam.services import tag_service as existing_tag_service

logger = logging.getLogger(__name__)

# Identifier for tagging models within ModelExecutionManager
TAGGING_MODEL_IDENTIFIER = "image_tagger"


# Registry for conceptual parameters of tagging models (not for loading, but for behavior)
TAGGING_MODEL_CONCEPTUAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "wd-v1-4-moat-tagger-v2": {
        "default_conceptual_params": {"threshold": 0.35, "tag_limit": 50},
        # "model_load_params": {} # These would be passed to ModelExecutionManager.get_model's `params`
    },
    # "another-tagger-v1": { ... }
}


# Module-level exports for direct use if needed, and for __init__.py
__all__ = [
    "TAGGING_MODEL_CONCEPTUAL_PARAMS",
]
