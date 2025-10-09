"""Provides functions for image tagging using Sire resources."""

import logging
from typing import Any

from dam.models.tags import (
    ModelGeneratedTagLinkComponent,
)
from dam_sire.resource import SireResource
from sire.core.runtime_resource_management import AutoManageWrapper
from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

TAGGING_MODEL_IDENTIFIER = "image_tagger"


class MockWd14Tagger:
    """A mock class for a WD14 Tagger model."""

    def __init__(self, model_name_or_path: str, params: dict[str, Any] | None = None):
        """Initialize the mock tagger."""
        self.model_name = model_name_or_path
        self.params = params if params else {}
        logger.info("MockWd14Tagger '%s' initialized with params %s", self.model_name, self.params)

    def predict(self, image_path: str, threshold: float, tag_limit: int) -> list[dict[str, Any]]:
        """Generate mock predictions based on the image path."""
        logger.info(
            "MockWd14Tagger '%s' predicting for %s with threshold %s, limit %s",
            self.model_name,
            image_path,
            threshold,
            tag_limit,
        )
        if "cat" in image_path.lower():
            return [
                {"tag_name": "animal", "confidence": 0.95},
                {"tag_name": "cat", "confidence": 0.92},
                {"tag_name": "pet", "confidence": 0.88},
            ]
        if "dog" in image_path.lower():
            return [
                {"tag_name": "animal", "confidence": 0.96},
                {"tag_name": "dog", "confidence": 0.93},
                {"tag_name": "pet", "confidence": 0.89},
                {"tag_name": "canine", "confidence": 0.75},
            ]
        return [{"tag_name": "generic_tag1", "confidence": 0.80}, {"tag_name": "generic_tag2", "confidence": 0.70}]


TAGGING_MODEL_CONCEPTUAL_PARAMS: dict[str, dict[str, Any]] = {
    "wd-v1-4-moat-tagger-v2": {
        "default_conceptual_params": {"threshold": 0.35, "tag_limit": 50},
    },
}


async def get_tagging_model(
    sire_resource: "SireResource",
    model_name: str,
    model_load_params: dict[str, Any] | None = None,
) -> Any | None:
    """Get a tagging model from the Sire resource manager."""
    AutoManageWrapper.registe_type_wrapper(MockWd14Tagger, TorchModuleWrapper)  # type: ignore
    return sire_resource.get_model(MockWd14Tagger, model_name, params=model_load_params)  # type: ignore


async def generate_tags_from_image(
    _sire_resource: "SireResource",
    _image_path: str,
    _model_name: str,
    _conceptual_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate tags for an image using a specified model."""
    logger.warning("generate_tags_from_image is not fully implemented with sire yet.")
    return []


async def update_entity_model_tags(
    _session: AsyncSession,
    _sire_resource: "SireResource",
    _entity_id: int,
    _image_path: str,
    _model_name: str,
) -> list[ModelGeneratedTagLinkComponent]:
    """Generate tags for an image and update the corresponding entity in the database."""
    logger.warning("update_entity_model_tags is not fully implemented with sire yet.")
    return []


__all__ = [
    "TAGGING_MODEL_CONCEPTUAL_PARAMS",
    "generate_tags_from_image",
    "get_tagging_model",
    "update_entity_model_tags",
]
