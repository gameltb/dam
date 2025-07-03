import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, FrozenSet

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam.core import get_default_world # To get ModelExecutionManager resource
from dam.core.model_manager import ModelExecutionManager
from dam.models.core.entity import Entity
from dam.models.semantic.audio_embedding_component import (
    get_audio_embedding_component_class,
    BaseSpecificAudioEmbeddingComponent,
    AudioModelHyperparameters,
    AUDIO_EMBEDDING_MODEL_REGISTRY,
)
from dam.services import ecs_service
from dam.utils.media_utils import get_file_path_for_entity

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_MODEL_NAME = "vggish" # This should match a key in AUDIO_EMBEDDING_MODEL_REGISTRY
MOCK_AUDIO_MODEL_IDENTIFIER = "mock_audio_model" # Identifier for ModelExecutionManager

# This class remains as it defines the mock model's behavior
class MockAudioModel:
    def __init__(self, model_name: str, params: Optional[AudioModelHyperparameters]):
        self.model_name = model_name
        self.params = params
        self.output_dim = AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name, {}).get("default_params", {}).get("dimensions", 128)
        logger.info(f"MockAudioModel '{model_name}' initialized with output_dim: {self.output_dim}")

    def encode(self, audio_path: str, **kwargs) -> np.ndarray:
        logger.info(f"MockAudioModel '{self.model_name}': Simulating encoding for '{audio_path}'")
        # Simulate some processing time
        # await asyncio.sleep(0.1)
        # Return a random numpy array of the expected dimension
        return np.random.rand(self.output_dim).astype(np.float32)

    async def encode_async(self, audio_path: str, **kwargs) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, audio_path, **kwargs)

# This is the loader function that ModelExecutionManager will use.
def _load_mock_audio_model_sync(model_name_or_path: str, params: Optional[Dict[str, Any]] = None) -> MockAudioModel:
    """
    Synchronous helper to load/create a MockAudioModel.
    `model_name_or_path` is used as `model_name` for MockAudioModel.
    `params` are passed to MockAudioModel constructor.
    """
    logger.info(f"Attempting to load MockAudioModel: {model_name_or_path} with params {params}")
    # model_name_or_path here corresponds to the conceptual model_name like "vggish"
    return MockAudioModel(model_name_or_path, params)

async def get_mock_audio_model(
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    params: Optional[AudioModelHyperparameters] = None,
    world_name: Optional[str] = None,
) -> MockAudioModel:
    """
    Loads a MockAudioModel using the ModelExecutionManager.
    """
    world = get_default_world()
    if world_name:
        from dam.core import get_world
        _w = get_world(world_name)
        if _w: world = _w
        else: logger.error(f"World {world_name} not found for audio model. Using default.")

    if not world:
        raise RuntimeError("Default world not found, cannot access ModelExecutionManager for audio models.")

    model_manager = world.get_resource(ModelExecutionManager)

    if MOCK_AUDIO_MODEL_IDENTIFIER not in model_manager._model_loaders:
        model_manager.register_model_loader(MOCK_AUDIO_MODEL_IDENTIFIER, _load_mock_audio_model_sync)

    # Ensure conceptual params from AUDIO_EMBEDDING_MODEL_REGISTRY are used if no specific params given
    loader_params = params
    if loader_params is None:
        registry_entry = AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            loader_params = registry_entry.get("default_params", {})
        else:
            loader_params = {}
            logger.warning(f"Audio model {model_name} not in registry, using empty params for MockAudioModel.")

    # Add device preference to loader_params if not already specified by caller
    if "device" not in loader_params: # Mock model doesn't use device, but good practice
        loader_params["device"] = model_manager.get_model_device_preference()


    # model_name for MockAudioModel is passed as model_name_or_path to manager.get_model
    return await model_manager.get_model(
        model_identifier=MOCK_AUDIO_MODEL_IDENTIFIER,
        model_name_or_path=model_name, # This is "vggish", "panns_cnn14", etc.
        params=loader_params
    )

def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


async def generate_audio_embedding_for_entity(
    session: AsyncSession,
    entity_id: int,
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    model_params: Optional[AudioModelHyperparameters] = None,
    audio_file_path: Optional[str] = None, # Allow passing path directly, e.g. during ingestion
    world_name: Optional[str] = None, # For ModelExecutionManager context
) -> Optional[BaseSpecificAudioEmbeddingComponent]:
    """
    Generates and stores a specific AudioEmbeddingComponent for an entity's audio file.
    Uses the registry to determine the correct table/component class.
    """
    AudioEmbeddingComponentClass = get_audio_embedding_component_class(model_name, model_params)
    if not AudioEmbeddingComponentClass:
        logger.error(
            f"No audio embedding component class found for model '{model_name}' and params {model_params}. Skipping embedding generation."
        )
        return None

    if model_params is None:
        registry_entry = AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            model_params = registry_entry.get("default_params", {})
        else:
            model_params = {}

    if audio_file_path is None:
        # Attempt to get the file path from the entity's FileLocationComponent
        # This assumes FileLocationComponent and a utility to resolve its path exist.
        # Replace with actual logic to get file path.
        # For now, using a placeholder from dam.utils.media_utils (which might need creation/update)
        try:
            # This utility needs to be robust and correctly locate the primary audio file for an entity.
            # It might involve querying for FileLocationComponent and constructing the full path.
            # file_path = await get_file_path_for_entity(session, entity_id, media_type="audio")
            # Placeholder:
            # For the purpose of this service structure, we'll assume a path can be found or is given.
            # In a real scenario, this part is critical.
            # Let's assume for now it's passed or we mock it.
            # If we can't get a path, we can't proceed.
            # This function might need to be part of a resource that has access to storage configuration.
            # For example: file_storage_resource.get_entity_file_path(entity_id, preferred_variant="original_audio")
            logger.warning(f"audio_file_path not provided for entity {entity_id}, and dynamic path retrieval is not fully implemented. Attempting mock path.")
            # This is a placeholder. The actual file path retrieval needs to be implemented.
            # It should likely use a FileStorageResource or similar to get the path to the asset.
            # For testing, we can assume it's passed or use a known dummy path if this service is called directly.
            # If this is called from a system processing an asset, the system might already have the path.
            # Let's make audio_file_path mandatory for now if not using get_file_path_for_entity
            raise FileNotFoundError(f"Audio file path must be provided or resolvable for entity {entity_id}")

        except FileNotFoundError as e:
            logger.error(f"Could not find audio file for entity {entity_id} to generate embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Error obtaining audio file path for entity {entity_id}: {e}")
            return None
    else:
        file_path = audio_file_path


    logger.info(f"Generating audio embedding for entity {entity_id} using model {model_name} from file: {file_path}")

    try:
        # Pass world_name for context
        model_instance = await get_mock_audio_model(model_name, model_params, world_name=world_name)
        # In a real scenario, this would involve loading the audio, preprocessing, and then encoding.
        # The mock model just needs the path for logging.

        # Example of how batching could be influenced by ModelExecutionManager (conceptual)
        # world = get_default_world(world_name)
        # model_manager = world.get_resource(ModelExecutionManager)
        # item_size_estimate_mb = 50 # Estimate for a typical audio file tensor after processing
        # batch_size = model_manager.get_optimal_batch_size(MOCK_AUDIO_MODEL_IDENTIFIER, model_name, item_size_estimate_mb)
        # For single file processing, batch_size is 1. If processing multiple, this would be relevant.

        embedding_np = await model_instance.encode_async(file_path)
        if embedding_np is None:
            logger.error(f"Audio embedding generation returned None for entity {entity_id}, file {file_path}")
            return None
    except Exception as e:
        logger.error(f"Audio embedding generation failed for entity {entity_id}, file {file_path}, model {model_name}: {e}", exc_info=True)
        return None

    embedding_bytes = convert_embedding_to_bytes(embedding_np)

    # Check for existing embedding for this entity and model
    # Note: BaseSpecificAudioEmbeddingComponent has model_name as a column
    existing_embeddings = await ecs_service.get_components_by_value(
        session,
        entity_id,
        AudioEmbeddingComponentClass,
        attributes_values={"model_name": model_name}, # Query by model_name within the specific table
    )

    emb_comp: BaseSpecificAudioEmbeddingComponent
    if existing_embeddings:
        emb_comp = existing_embeddings[0]
        if emb_comp.embedding_vector != embedding_bytes:
            emb_comp.embedding_vector = embedding_bytes
            # emb_comp.model_name = model_name # Already set, and table implies model type
            session.add(emb_comp)
            logger.info(
                f"Updated {AudioEmbeddingComponentClass.__name__} for entity {entity_id}, model: {model_name}"
            )
        else:
            logger.debug(
                f"{AudioEmbeddingComponentClass.__name__} for entity {entity_id}, model: {model_name} is up-to-date."
            )
    else:
        emb_comp = AudioEmbeddingComponentClass(
            embedding_vector=embedding_bytes,
            model_name=model_name, # Explicitly set model_name
        )
        await ecs_service.add_component_to_entity(session, entity_id, emb_comp)
        logger.info(
            f"Created {AudioEmbeddingComponentClass.__name__} for entity {entity_id}, model: {model_name}"
        )

    await session.flush() # Ensure component ID is populated if new
    return emb_comp


async def get_audio_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    model_name: str,
    model_params: Optional[AudioModelHyperparameters] = None,
) -> List[BaseSpecificAudioEmbeddingComponent]:
    """
    Retrieves specific AudioEmbeddingComponents for an entity, from the table
    corresponding to model_name and filtered by the model_name column.
    """
    AudioEmbeddingComponentClass = get_audio_embedding_component_class(model_name, model_params)
    if not AudioEmbeddingComponentClass:
        logger.warning(f"No component class for audio model {model_name}, params {model_params}. Cannot get embeddings.")
        return []

    # Filter by model_name as it's a column in BaseSpecificAudioEmbeddingComponent
    return await ecs_service.get_components_by_value(
        session,
        entity_id,
        AudioEmbeddingComponentClass,
        attributes_values={"model_name": model_name}
    )


async def find_similar_entities_by_audio_embedding(
    session: AsyncSession,
    query_audio_path: str, # Path to the query audio file
    model_name: str,
    model_params: Optional[AudioModelHyperparameters] = None,
    top_n: int = 10,
) -> List[Tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]]:
    """
    Finds entities with similar audio content to the query audio file using cosine similarity
    on stored audio embeddings from the table specified by model_name.
    """
    AudioEmbeddingComponentClass = get_audio_embedding_component_class(model_name, model_params)
    if not AudioEmbeddingComponentClass:
        logger.error(f"No component class for audio model {model_name}, params {model_params}. Cannot find similar entities.")
        return []

    if model_params is None:
        registry_entry = AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            model_params = registry_entry.get("default_params", {})
        else:
            model_params = {}

    try:
        model_instance = await get_mock_audio_model(model_name, model_params)
        query_embedding_np = await model_instance.encode_async(query_audio_path)
        if query_embedding_np is None:
            logger.error(f"Could not generate query audio embedding for '{query_audio_path}' with model {model_name}.")
            return []
    except Exception as e:
        logger.error(f"Failed to generate query audio embedding for '{query_audio_path}': {e}", exc_info=True)
        return []


    # Fetch all relevant embeddings from the specific table, filtering by model_name
    all_embedding_components_stmt = select(AudioEmbeddingComponentClass).where(
        AudioEmbeddingComponentClass.model_name == model_name
    )
    result = await session.execute(all_embedding_components_stmt)
    all_embedding_components = result.scalars().all()

    if not all_embedding_components:
        logger.info(f"No audio embeddings found in table for {AudioEmbeddingComponentClass.__name__} with model_name '{model_name}' to compare against.")
        return []

    similarities = []
    for emb_comp in all_embedding_components:
        db_embedding_np = convert_bytes_to_embedding(emb_comp.embedding_vector)
        dot_product = np.dot(query_embedding_np, db_embedding_np)
        norm_query = np.linalg.norm(query_embedding_np)
        norm_db = np.linalg.norm(db_embedding_np)

        if norm_query == 0 or norm_db == 0:
            score = 0.0
        else:
            score = dot_product / (norm_query * norm_db)
        similarities.append((emb_comp.entity_id, score, emb_comp))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_results: List[Tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]] = []
    entity_ids_processed = set()
    for entity_id, score, emb_comp_instance in similarities:
        if len(top_results) >= top_n:
            break
        # Allow multiple results from the same entity if they have different embeddings
        # (e.g. from different segments, though not implemented yet)
        # For now, assume one embedding per entity-model.
        if entity_id in entity_ids_processed:
            continue


        entity = await ecs_service.get_entity(session, entity_id)
        if entity:
            top_results.append((entity, float(score), emb_comp_instance))
            entity_ids_processed.add(entity_id)
            if len(entity_ids_processed) >=top_n : # if we only want unique entities
                 break


    return top_results

__all__ = [
    "generate_audio_embedding_for_entity",
    "get_audio_embeddings_for_entity",
    "find_similar_entities_by_audio_embedding",
    "DEFAULT_AUDIO_MODEL_NAME",
    "get_mock_audio_model", # Exporting for potential direct use/testing
]
