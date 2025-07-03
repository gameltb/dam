import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from dam.models.core import Entity
from dam.models.tags import (
    TagConceptComponent,
    EntityTagLinkComponent, # For manual tags
    ModelGeneratedTagLinkComponent, # For AI model generated tags
)
from dam.services import ecs_service, tag_service as existing_tag_service # Assuming tag_service handles TagConcept

logger = logging.getLogger(__name__)

# Placeholder type for a model loading function
TagModelLoaderFn = Callable[[Dict[str, Any]], Any] # Takes params, returns loaded model

# Registry for tagging models
TAGGING_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wd-v1-4-moat-tagger-v2": {
        # "loader_function": load_wd14_tagger, # Actual function to load this model
        "model_instance": None, # Will be populated by get_tagging_model
        "default_conceptual_params": {"threshold": 0.35, "tag_limit": 50}, # Conceptual params
        "model_load_params": {}, # Params for the loader_function itself
    },
    # Example for another model
    # "another-tagger-v1": {
    #     "loader_function": load_another_tagger,
    #     "model_instance": None,
    #     "default_conceptual_params": {"min_confidence": 0.5},
    #     "model_load_params": {"device": "cpu"},
    # }
}

async def get_tagging_model(model_name: str) -> Optional[Any]:
    """
    Loads and caches a tagging model from the registry.
    For now, this is a simplified loader. Real implementation would involve
    calling the specific loader_function.
    """
    registry_entry = TAGGING_MODEL_REGISTRY.get(model_name)
    if not registry_entry:
        logger.error(f"Tagging model '{model_name}' not found in registry.")
        return None

    if registry_entry.get("model_instance") is None:
        logger.info(f"Loading tagging model: {model_name}")
        # In a real scenario, you would call:
        # loader_fn = registry_entry.get("loader_function")
        # if loader_fn:
        #     try:
        #         # This might need to be async or run in executor if loading is slow
        #         registry_entry["model_instance"] = loader_fn(registry_entry.get("model_load_params", {}))
        #         logger.info(f"Tagging model {model_name} loaded.")
        #     except Exception as e:
        #         logger.error(f"Failed to load tagging model {model_name}: {e}", exc_info=True)
        #         return None
        # else:
        #     logger.error(f"No loader_function defined for tagging model {model_name}")
        #     return None

        # --- MOCK IMPLEMENTATION for wd-v1-4-moat-tagger-v2 ---
        if model_name == "wd-v1-4-moat-tagger-v2":
            # Simulate a loaded model object with an 'predict' or 'generate_tags' method
            class MockWd14Tagger:
                def predict(self, image_path: str, threshold: float, tag_limit: int):
                    logger.info(f"MockWd14Tagger predicting for {image_path} with threshold {threshold}, limit {tag_limit}")
                    # Simulate some tags based on image_path for predictability in tests
                    if "cat" in image_path.lower():
                        return [{"tag_name": "animal", "confidence": 0.95}, {"tag_name": "cat", "confidence": 0.92}, {"tag_name": "pet", "confidence": 0.88}]
                    elif "dog" in image_path.lower():
                         return [{"tag_name": "animal", "confidence": 0.96}, {"tag_name": "dog", "confidence": 0.93}, {"tag_name": "pet", "confidence": 0.89}, {"tag_name": "canine", "confidence": 0.75}]
                    else:
                        return [{"tag_name": "generic_tag1", "confidence": 0.80}, {"tag_name": "generic_tag2", "confidence": 0.70}]
            registry_entry["model_instance"] = MockWd14Tagger()
            logger.info(f"Mock tagging model {model_name} initialized.")
        else:
            logger.error(f"No mock implementation available for {model_name} and no loader_function defined.")
            return None
        # --- END MOCK IMPLEMENTATION ---

    return registry_entry["model_instance"]


async def generate_tags_from_image(
    image_path: str, # In real system, this might be bytes or a more abstract FileObject
    model_name: str,
    conceptual_params: Dict[str, Any], # e.g., threshold, tag_limit
) -> List[Dict[str, Any]]: # List of {"tag_name": str, "confidence": float}
    """
    Generates tags for a given image using the specified model and parameters.
    """
    model = await get_tagging_model(model_name)
    if not model:
        return []

    try:
        # This is where the model-specific prediction logic would go.
        # The mock model has a 'predict' method.
        # Real models might have different APIs.
        if hasattr(model, "predict"):
            # Pass relevant conceptual_params to the model's predict method
            # This requires knowing what the mock/real model expects.
            raw_tags = model.predict(
                image_path,
                threshold=conceptual_params.get("threshold", 0.1), # Example
                tag_limit=conceptual_params.get("tag_limit", 100) # Example
            )
            # Ensure output format matches List[Dict[str, Any]]
            # For the mock, it already matches. Real models might need transformation.
            return raw_tags
        else:
            logger.error(f"Model {model_name} does not have a 'predict' method as expected by this service.")
            return []
    except Exception as e:
        logger.error(f"Error generating tags with model {model_name} for image {image_path}: {e}", exc_info=True)
        return []


async def update_entity_model_tags(
    session: AsyncSession,
    entity_id: int,
    image_path: str, # Path to the image file to be tagged
    model_name: str,
    # model_conceptual_params_override: Optional[Dict[str, Any]] = None, # For overriding registry defaults
) -> List[ModelGeneratedTagLinkComponent]:
    """
    Updates the AI-generated tags for an entity using a specified model.
    It will remove all existing tags from this specific model for the entity
    and then add the newly generated ones.
    """
    registry_entry = TAGGING_MODEL_REGISTRY.get(model_name)
    if not registry_entry:
        logger.error(f"Cannot update tags: Model '{model_name}' not found in registry.")
        return []

    # Get conceptual parameters (e.g., confidence threshold for filtering tags)
    # conceptual_params = registry_entry.get("default_conceptual_params", {}).copy()
    # if model_conceptual_params_override:
    #     conceptual_params.update(model_conceptual_params_override)
    # For now, just use defaults from registry for simplicity
    conceptual_params = registry_entry.get("default_conceptual_params", {})


    # 1. Generate new tags using the model
    generated_raw_tags = await generate_tags_from_image(image_path, model_name, conceptual_params)

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

        # Optional: Apply a conceptual threshold here if the model itself doesn't do it
        # For example, if conceptual_params has a "min_store_confidence"
        # min_conf_to_store = conceptual_params.get("min_store_confidence", 0.0)
        # if confidence is not None and confidence < min_conf_to_store:
        #    continue

        # Get or create the canonical TagConceptComponent
        tag_concept = await existing_tag_service.get_or_create_tag_concept(session, tag_name.strip().lower())
        if not tag_concept:
            logger.error(f"Could not get or create TagConceptComponent for tag: {tag_name}")
            continue

        # Create the ModelGeneratedTagLinkComponent
        model_tag_link = ModelGeneratedTagLinkComponent(
            entity_id=entity_id,
            tag_concept_id=tag_concept.id,
            source_model_name=model_name,
            confidence=float(confidence) if confidence is not None else None,
        )
        session.add(model_tag_link)
        created_tag_links.append(model_tag_link)

    if created_tag_links:
        # The calling function/system should handle session.flush() or commit()
        logger.info(f"Added/Updated {len(created_tag_links)} tags from model '{model_name}' for entity {entity_id}.")

    return created_tag_links

# TODO: Add functions to retrieve model-generated tags for an entity, if needed.
# async def get_model_generated_tags_for_entity(
#     session: AsyncSession,
#     entity_id: int,
#     model_name: Optional[str] = None # Filter by specific model if provided
# ) -> List[ModelGeneratedTagLinkComponent]:
#     stmt = select(ModelGeneratedTagLinkComponent).where(ModelGeneratedTagLinkComponent.entity_id == entity_id)
#     if model_name:
#         stmt = stmt.where(ModelGeneratedTagLinkComponent.source_model_name == model_name)
#     result = await session.execute(stmt)
#     return list(result.scalars().all())
