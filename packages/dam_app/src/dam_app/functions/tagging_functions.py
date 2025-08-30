import logging
from typing import Any, Dict, List, Optional

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
    def __init__(self, model_name_or_path: str, params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name_or_path
        self.params = params if params else {}
        logger.info(f"MockWd14Tagger '{self.model_name}' initialized with params {self.params}")

    def predict(self, image_path: str, threshold: float, tag_limit: int) -> List[Dict[str, Any]]:
        logger.info(
            f"MockWd14Tagger '{self.model_name}' predicting for {image_path} with threshold {threshold}, limit {tag_limit}"
        )
        if "cat" in image_path.lower():
            return [
                {"tag_name": "animal", "confidence": 0.95},
                {"tag_name": "cat", "confidence": 0.92},
                {"tag_name": "pet", "confidence": 0.88},
            ]
        elif "dog" in image_path.lower():
            return [
                {"tag_name": "animal", "confidence": 0.96},
                {"tag_name": "dog", "confidence": 0.93},
                {"tag_name": "pet", "confidence": 0.89},
                {"tag_name": "canine", "confidence": 0.75},
            ]
        else:
            return [{"tag_name": "generic_tag1", "confidence": 0.80}, {"tag_name": "generic_tag2", "confidence": 0.70}]


def _load_mock_tagging_model_sync(model_name_or_path: str, params: Optional[Dict[str, Any]] = None) -> MockWd14Tagger:
    return MockWd14Tagger(model_name_or_path, params)


TAGGING_MODEL_CONCEPTUAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "wd-v1-4-moat-tagger-v2": {
        "default_conceptual_params": {"threshold": 0.35, "tag_limit": 50},
    },
}


async def get_tagging_model(
    sire_resource: "SireResource",
    model_name: str,
    model_load_params: Optional[Dict[str, Any]] = None,
) -> Optional["AutoManageWrapper"]:
    AutoManageWrapper.registe_type_wrapper(MockWd14Tagger, TorchModuleWrapper)
    return sire_resource.get_model(MockWd14Tagger, model_name, params=model_load_params)


async def generate_tags_from_image(
    sire_resource: "SireResource",
    image_path: str,
    model_name: str,
    conceptual_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    logger.warning("generate_tags_from_image is not fully implemented with sire yet.")
    return []


async def update_entity_model_tags(
    session: AsyncSession,
    sire_resource: "SireResource",
    entity_id: int,
    image_path: str,
    model_name: str,
) -> List[ModelGeneratedTagLinkComponent]:
    logger.warning("update_entity_model_tags is not fully implemented with sire yet.")
    return []


__all__ = [
    "TAGGING_MODEL_CONCEPTUAL_PARAMS",
    "get_tagging_model",
    "generate_tags_from_image",
    "update_entity_model_tags",
]
