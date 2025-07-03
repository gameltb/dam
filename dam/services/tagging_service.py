import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from dam.core import get_default_world
from dam.core.model_manager import ModelExecutionManager
from dam.models.core.entity import Entity
from dam.models.tags import (
    TagConceptComponent,
    ModelGeneratedTagLinkComponent,
)
from dam.services import ecs_service, tag_service as existing_tag_service

logger = logging.getLogger(__name__)

# Identifier for tagging models within ModelExecutionManager
TAGGING_MODEL_IDENTIFIER = "image_tagger"

# --- Mock Model Definition (kept for this refactoring) ---
class MockWd14Tagger:
    def __init__(self, model_name_or_path: str, params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name_or_path
        self.params = params if params else {}
        logger.info(f"MockWd14Tagger '{self.model_name}' initialized with params {self.params}")

    def predict(self, image_path: str, threshold: float, tag_limit: int) -> List[Dict[str, Any]]:
        logger.info(f"MockWd14Tagger '{self.model_name}' predicting for {image_path} with threshold {threshold}, limit {tag_limit}")
        if "cat" in image_path.lower():
            return [{"tag_name": "animal", "confidence": 0.95}, {"tag_name": "cat", "confidence": 0.92}, {"tag_name": "pet", "confidence": 0.88}]
        elif "dog" in image_path.lower():
            return [{"tag_name": "animal", "confidence": 0.96}, {"tag_name": "dog", "confidence": 0.93}, {"tag_name": "pet", "confidence": 0.89}, {"tag_name": "canine", "confidence": 0.75}]
        else:
            return [{"tag_name": "generic_tag1", "confidence": 0.80}, {"tag_name": "generic_tag2", "confidence": 0.70}]

# --- Model Loader for ModelExecutionManager ---
def _load_mock_tagging_model_sync(model_name_or_path: str, params: Optional[Dict[str, Any]] = None) -> MockWd14Tagger:
    """
    Synchronous loader for the MockWd14Tagger.
    `model_name_or_path` can be used to differentiate mock behaviors if needed, or just for logging.
    `params` are passed to the mock model constructor.
    """
    # Here, model_name_or_path is the specific "model name" like "wd-v1-4-moat-tagger-v2"
    return MockWd14Tagger(model_name_or_path, params)


# Registry for conceptual parameters of tagging models (not for loading, but for behavior)
TAGGING_MODEL_CONCEPTUAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "wd-v1-4-moat-tagger-v2": {
        "default_conceptual_params": {"threshold": 0.35, "tag_limit": 50},
        # "model_load_params": {} # These would be passed to ModelExecutionManager.get_model's `params`
    },
    # "another-tagger-v1": { ... }
}


async def get_tagging_model(
    model_name: str, # e.g., "wd-v1-4-moat-tagger-v2"
    model_load_params: Optional[Dict[str, Any]] = None, # Params for actual loading (e.g. device)
    world_name: Optional[str] = None,
) -> Optional[MockWd14Tagger]: # Specific to mock for now
    """
    Loads a tagging model using the ModelExecutionManager.
    """
    world = get_default_world()
    if world_name:
        from dam.core import get_world as get_world_dyn
        _w = get_world_dyn(world_name)
        if _w: world = _w
        else: logger.error(f"World {world_name} not found for tagging model. Using default.")

    if not world:
        raise RuntimeError("Default world not found, cannot access ModelExecutionManager for tagging.")

    model_manager = world.get_resource(ModelExecutionManager)

    if TAGGING_MODEL_IDENTIFIER not in model_manager._model_loaders:
        model_manager.register_model_loader(TAGGING_MODEL_IDENTIFIER, _load_mock_tagging_model_sync)

    final_load_params = model_load_params.copy() if model_load_params else {}
    # Merge with any default load params from conceptual registry if they exist for this model_name
    conceptual_entry = TAGGING_MODEL_CONCEPTUAL_PARAMS.get(model_name, {})
    default_loader_params_for_model = conceptual_entry.get("model_load_params", {}) # type: ignore
    final_load_params = {**default_loader_params_for_model, **final_load_params}


    if "device" not in final_load_params:
        final_load_params["device"] = model_manager.get_model_device_preference()

    # The `model_name` (e.g., "wd-v1-4-moat-tagger-v2") is passed as `model_name_or_path`
    # to the ModelExecutionManager, which then passes it to the loader.
    return await model_manager.get_model(
        model_identifier=TAGGING_MODEL_IDENTIFIER,
        model_name_or_path=model_name,
        params=final_load_params
    )


async def generate_tags_from_image(
    image_path: str,
    model_name: str, # e.g., "wd-v1-4-moat-tagger-v2"
    conceptual_params: Dict[str, Any], # e.g., threshold, tag_limit for prediction behavior
    world_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generates tags for a given image using the specified model and parameters.
    """
    # model_load_params could be extracted from conceptual_params or TAGGING_MODEL_CONCEPTUAL_PARAMS
    # if some conceptual params also affect loading for this model type.
    # For now, assume model_load_params are empty or handled by defaults in get_tagging_model.
    model_load_params_from_conceptual = TAGGING_MODEL_CONCEPTUAL_PARAMS.get(model_name, {}).get("model_load_params")

    model = await get_tagging_model(model_name, model_load_params=model_load_params_from_conceptual, world_name=world_name) # type: ignore
    if not model:
        return []

    try:
        if hasattr(model, "predict"):
            raw_tags = model.predict(
                image_path,
                threshold=conceptual_params.get("threshold", 0.1),
                tag_limit=conceptual_params.get("tag_limit", 100)
            )
            return raw_tags
        else:
            logger.error(f"Model {model_name} does not have a 'predict' method.")
            return []
    except Exception as e:
        logger.error(f"Error generating tags with model {model_name} for image {image_path}: {e}", exc_info=True)
        return []


async def update_entity_model_tags(
    session: AsyncSession,
    entity_id: int,
    image_path: str,
    model_name: str, # e.g., "wd-v1-4-moat-tagger-v2"
    world_name: Optional[str] = None,
) -> List[ModelGeneratedTagLinkComponent]:
    """
    Updates the AI-generated tags for an entity using a specified model.
    It will remove all existing tags from this specific model for the entity
    and then add the newly generated ones.
    """
    conceptual_registry_entry = TAGGING_MODEL_CONCEPTUAL_PARAMS.get(model_name)
    if not conceptual_registry_entry:
        logger.error(f"Cannot update tags: Model '{model_name}' not found in TAGGING_MODEL_CONCEPTUAL_PARAMS for its conceptual params.")
        return []

    # Get conceptual parameters (e.g., confidence threshold for filtering tags)
    conceptual_params = conceptual_registry_entry.get("default_conceptual_params", {})


    # 1. Generate new tags using the model
    # Pass world_name to generate_tags_from_image
    generated_raw_tags = await generate_tags_from_image(image_path, model_name, conceptual_params, world_name=world_name)

    if not generated_raw_tags:
        logger.info(f"No tags generated by {model_name} for entity {entity_id} (image: {image_path}).")
        # Still proceed to delete old tags if any

    # 2. Delete existing tags from this model for this entity
    stmt_delete = delete(ModelGeneratedTagLinkComponent).where(
        ModelGeneratedTagLinkComponent.entity_id == entity_id,
        ModelGeneratedTagLinkComponent.source_model_name == model_name,
    )
    await session.execute(stmt_delete)
    logger.debug(f"Deleted existing tags from model '{model_name}' for entity {entity_id}.")

    # 3. Add new tags
    created_tag_links: List[ModelGeneratedTagLinkComponent] = []
    for raw_tag in generated_raw_tags:
        tag_name = raw_tag.get("tag_name")
        confidence = raw_tag.get("confidence")

        if not tag_name or not isinstance(tag_name, str):
            logger.warning(f"Skipping invalid tag from model {model_name}: {raw_tag}")
            continue

        tag_concept = await existing_tag_service.get_or_create_tag_concept(session, tag_name.strip().lower())
        if not tag_concept: # Should not happen if get_or_create always returns or raises
            logger.error(f"Could not get or create TagConceptComponent for tag: {tag_name}")
            continue

        model_tag_link = ModelGeneratedTagLinkComponent(
            entity_id=entity_id, # type: ignore
            tag_concept_id=tag_concept.id, # type: ignore
            source_model_name=model_name,
            confidence=float(confidence) if confidence is not None else None,
        )
        session.add(model_tag_link)
        created_tag_links.append(model_tag_link)

    if created_tag_links:
        logger.info(f"Added/Updated {len(created_tag_links)} tags from model '{model_name}' for entity {entity_id}.")

    return created_tag_links


class TaggingService:
    """
    A service class for tagging operations. This class itself can be registered
    as a resource. Its methods will wrap the module-level service functions.
    """
    async def update_entity_model_tags(
        self,
        session: AsyncSession,
        entity_id: int,
        image_path: str,
        model_name: str,
        world_name: Optional[str] = None,
    ) -> List[ModelGeneratedTagLinkComponent]:
        return await update_entity_model_tags(
            session, entity_id, image_path, model_name, world_name
        )

    async def generate_tags_from_image(
        self,
        image_path: str,
        model_name: str,
        conceptual_params: Dict[str, Any],
        world_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await generate_tags_from_image(
            image_path, model_name, conceptual_params, world_name
        )

    async def get_tagging_model(
        self,
        model_name: str,
        model_load_params: Optional[Dict[str, Any]] = None,
        world_name: Optional[str] = None,
    ) -> Optional[MockWd14Tagger]: # Specific to mock for now
        return await get_tagging_model(
            model_name, model_load_params, world_name # type: ignore
        )

# Module-level exports for direct use if needed, and for __init__.py
__all__ = [
    "generate_tags_from_image",
    "get_tagging_model",
    "TAGGING_MODEL_CONCEPTUAL_PARAMS",
    "update_entity_model_tags",
    "TaggingService", # Export the class
]
