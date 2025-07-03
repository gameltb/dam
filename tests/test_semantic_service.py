from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Models to be tested (actual specific component classes)
from dam.models.semantic import (
    TextEmbeddingAllMiniLML6V2Dim384Component,
    TextEmbeddingClipVitB32Dim512Component,
    get_embedding_component_class,
    EMBEDDING_MODEL_REGISTRY,
    ModelHyperparameters,
    BaseSpecificEmbeddingComponent,
)
from dam.services import ecs_service, semantic_service
from dam.services.semantic_service import BatchTextItem # Keep this for type hints

# Define model names and params to be used in tests, corresponding to registered models
TEST_MODEL_MINILM = "all-MiniLM-L6-v2"
TEST_PARAMS_MINILM: ModelHyperparameters = {"dimensions": 384} # Must match registry

TEST_MODEL_CLIP = "clip-ViT-B-32"
TEST_PARAMS_CLIP: ModelHyperparameters = {"dimensions": 512} # Must match registry

# Get the component classes for easier use in tests
MiniLMEmbeddingComponent = get_embedding_component_class(TEST_MODEL_MINILM, TEST_PARAMS_MINILM)
ClipEmbeddingComponent = get_embedding_component_class(TEST_MODEL_CLIP, TEST_PARAMS_CLIP)

# Ensure the test models are actually registered and component classes are found
assert MiniLMEmbeddingComponent is not None, "Test MiniLM model not registered or class not found"
assert ClipEmbeddingComponent is not None, "Test CLIP model not registered or class not found"


# Mock SentenceTransformer class (remains largely the same)
class MockSentenceTransformer:
    def __init__(self, model_name_or_path=None, **kwargs): # Added **kwargs to accept potential device args
        self.model_name = model_name_or_path
        # Determine embedding dimension based on model_name for more realistic mock
        if "clip" in model_name_or_path.lower():
            self.dim = 512
        else: # Default to MiniLM dimension
            self.dim = 384


    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        original_sentences_type = type(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for s in sentences:
            if not s or not s.strip():
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue
            sum_ords = sum(ord(c) for c in s)
            # Incorporate model name to make embeddings different for different models
            model_ord_sum = sum(ord(c) for c in self.model_name)
            vec = np.array([sum_ords % 100, len(s) % 100, model_ord_sum % 100] + [0.0] * (self.dim - 3), dtype=np.float32)
            embeddings.append(vec)

        if not convert_to_numpy:
            embeddings = [e.tolist() for e in embeddings]

        if original_sentences_type is str:
            return embeddings[0] if embeddings else np.array([])
        else:
            return np.array(embeddings) if convert_to_numpy else embeddings


@pytest.fixture(autouse=True)
def clear_model_cache_and_mock_load_sync():
    """Clears the model cache before and after each test, and mocks _load_model_sync."""
    semantic_service._model_cache.clear()
    # Patch _load_model_sync within semantic_service to use MockSentenceTransformer
    # The mock_load_model_sync_func will be called instead of the real _load_model_sync
    def mock_load_model_sync_func(model_name_str: str, model_load_params: Optional[Dict[str, Any]] = None):
        # print(f"Mock _load_model_sync called with: {model_name_str}, {model_load_params}")
        return MockSentenceTransformer(model_name_or_path=model_name_str)

    with patch("dam.services.semantic_service._load_model_sync", side_effect=mock_load_model_sync_func) as mock_loader:
        yield mock_loader # Provide the mock to the test if needed, though usually not directly
    semantic_service._model_cache.clear()


@pytest.mark.asyncio
async def test_generate_embedding_and_conversion():
    text = "Hello world"
    # Test with MiniLM
    embedding_minilm_np = await semantic_service.generate_embedding(
        text, model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
    )
    assert embedding_minilm_np is not None
    assert isinstance(embedding_minilm_np, np.ndarray)
    assert embedding_minilm_np.shape == (TEST_PARAMS_MINILM["dimensions"],)

    # Test with CLIP
    embedding_clip_np = await semantic_service.generate_embedding(
        text, model_name=TEST_MODEL_CLIP, params=TEST_PARAMS_CLIP
    )
    assert embedding_clip_np is not None
    assert isinstance(embedding_clip_np, np.ndarray)
    assert embedding_clip_np.shape == (TEST_PARAMS_CLIP["dimensions"],)

    # Ensure embeddings are different for different models
    # Due to mock's design, if text is same, first few values might be same if model_name effect is small.
    # A full array_equal check is better.
    # Reshape minilm to be comparable if dimensions were different (not in this mock's case for these two models if we check first few elements)
    # For now, we trust the mock's dim parameter.
    # A simple check: their sum should be different due to model_ord_sum in mock
    assert not np.array_equal(embedding_minilm_np, embedding_clip_np)


    # Test empty text
    assert await semantic_service.generate_embedding("", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM) is None
    assert await semantic_service.generate_embedding("   ", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM) is None

    # Test conversion (using MiniLM embedding)
    embedding_bytes = semantic_service.convert_embedding_to_bytes(embedding_minilm_np)
    assert isinstance(embedding_bytes, bytes)
    assert len(embedding_bytes) == TEST_PARAMS_MINILM["dimensions"] * 4 # 4 bytes per float32

    embedding_np_restored = semantic_service.convert_bytes_to_embedding(embedding_bytes)
    assert np.array_equal(embedding_minilm_np, embedding_np_restored)

    # Test with unregistered model name (should fail to get component class)
    # generate_embedding itself doesn't directly use get_embedding_component_class,
    # it uses get_sentence_transformer_model. So this call might still "succeed" at generating
    # if the mock loader can handle "unregistered-model".
    # The failure would occur in update_text_embeddings_for_entity.
    # However, get_sentence_transformer_model now uses registry for default params, so it might log warning.
    unregistered_embedding = await semantic_service.generate_embedding("test", model_name="unregistered-model", params={})
    # Depending on mock and error handling, this might return an embedding or raise.
    # MockSentenceTransformer will be created.
    assert unregistered_embedding is not None


@pytest.mark.asyncio
async def test_update_text_embeddings_for_entity_specific_tables(db_session: AsyncSession):
    entity = await ecs_service.create_entity(db_session)
    text_map = {
        "File.name": "document.pdf",
        "Desc.text": "Sample content.",
    }

    # Update with MiniLM model
    created_minilm = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    assert len(created_minilm) == 2
    assert all(isinstance(c, MiniLMEmbeddingComponent) for c in created_minilm)

    db_minilm_comps = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
    assert len(db_minilm_comps) == 2
    for comp in db_minilm_comps:
        assert comp.embedding_vector is not None
        source_key = f"{comp.source_component_name}.{comp.source_field_name}"
        original_text = text_map[source_key]
        expected_emb = await semantic_service.generate_embedding(
            original_text, model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb)

    # Update with CLIP model (should go to a different table)
    created_clip = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map, model_name=TEST_MODEL_CLIP, model_params=TEST_PARAMS_CLIP
    )
    assert len(created_clip) == 2
    assert all(isinstance(c, ClipEmbeddingComponent) for c in created_clip)

    db_clip_comps = await ecs_service.get_components(db_session, entity.id, ClipEmbeddingComponent)
    assert len(db_clip_comps) == 2
    for comp in db_clip_comps:
        assert comp.embedding_vector is not None
        source_key = f"{comp.source_component_name}.{comp.source_field_name}"
        original_text = text_map[source_key]
        expected_emb = await semantic_service.generate_embedding(
            original_text, model_name=TEST_MODEL_CLIP, params=TEST_PARAMS_CLIP
        )
        assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb)

    # Ensure MiniLM table still has its original 2 components
    db_minilm_comps_after_clip = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
    assert len(db_minilm_comps_after_clip) == 2

    # Test update (text change) for one model
    text_map_updated = {"File.name": "new_document.pdf", "Desc.text": "Sample content."}
    updated_minilm = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map_updated, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    assert len(updated_minilm) == 2
    db_minilm_updated = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
    assert len(db_minilm_updated) == 2 # Still 2 components
    for comp in db_minilm_updated:
        if comp.source_component_name == "File":
            expected_emb_updated = await semantic_service.generate_embedding(
                "new_document.pdf", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
            )
            assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb_updated)
            original_emb_old = await semantic_service.generate_embedding(
                "document.pdf", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
            )
            assert not np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), original_emb_old)

    # Test with an unregistered model - should return empty list and log error
    unregistered_results = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map, model_name="unregistered-model", model_params={}
    )
    assert len(unregistered_results) == 0


@pytest.mark.asyncio
async def test_get_text_embeddings_for_entity(db_session: AsyncSession):
    entity = await ecs_service.create_entity(db_session)
    text_map = {"CompA.field1": "Text A", "CompB.field2": "Text B"}

    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, {"CompC.field3": "Text C"}, model_name=TEST_MODEL_CLIP, model_params=TEST_PARAMS_CLIP
    )

    # Get MiniLM embeddings
    minilm_embs = await semantic_service.get_text_embeddings_for_entity(
        db_session, entity.id, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    assert len(minilm_embs) == 2
    assert all(isinstance(c, MiniLMEmbeddingComponent) for c in minilm_embs)

    # Get CLIP embeddings
    clip_embs = await semantic_service.get_text_embeddings_for_entity(
        db_session, entity.id, model_name=TEST_MODEL_CLIP, model_params=TEST_PARAMS_CLIP
    )
    assert len(clip_embs) == 1
    assert isinstance(clip_embs[0], ClipEmbeddingComponent)
    assert clip_embs[0].source_component_name == "CompC"

    # Get for non-existent model
    no_model_embs = await semantic_service.get_text_embeddings_for_entity(
        db_session, entity.id, model_name="non-existent", model_params={}
    )
    assert len(no_model_embs) == 0


@pytest.mark.asyncio
async def test_find_similar_entities_by_text_embedding(db_session: AsyncSession):
    entity1 = await ecs_service.create_entity(db_session) # Target for MiniLM
    entity2 = await ecs_service.create_entity(db_session) # Target for MiniLM
    entity3 = await ecs_service.create_entity(db_session) # Target for CLIP

    # Populate with MiniLM embeddings
    # Mock embedding: np.array([sum_ords % 100, len(s) % 100, model_ord_sum % 100] + [0.0] * (dim - 3))
    # model_ord_sum for "all-MiniLM-L6-v2" = 1841 % 100 = 41
    # Text1: "apple pie" -> sum_ords=826, len=9. vec1_minilm = [26, 9, 41, 0...]
    # Text2: "apple crumble" -> sum_ords=1230, len=13. vec2_minilm = [30, 13, 41, 0...]
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity1.id, {"Data.text": "apple pie"}, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity2.id, {"Data.text": "apple crumble"}, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )

    # Populate with CLIP embeddings (should not interfere with MiniLM search)
    # model_ord_sum for "clip-ViT-B-32" = 1230 % 100 = 30
    # Text3: "apple pie" (same text, different model) -> vec3_clip = [26, 9, 30, 0...]
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity3.id, {"Data.text": "apple pie"}, model_name=TEST_MODEL_CLIP, model_params=TEST_PARAMS_CLIP
    )

    # Query using MiniLM
    query_text = "apple pie recipe" # sum_ords=1528, len=16. vec_query_minilm = [28, 16, 41, 0...]
                                    # Using mock: sum_ords=1528 -> 28, len=16 -> 16.
                                    # model_ord_sum for "all-MiniLM-L6-v2" is 41.
                                    # Query vec: [28, 16, 41, ...]
                                    # e1 (apple pie): [26, 9, 41, ...]
                                    # e2 (apple crumble): [30, 13, 41, ...]
                                    # Cosine sim: query vs e1 should be higher than query vs e2.

    similar_minilm = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM, top_n=2
    )
    assert len(similar_minilm) == 2
    assert similar_minilm[0][0].id == entity1.id # apple pie
    assert similar_minilm[1][0].id == entity2.id # apple crumble
    assert similar_minilm[0][1] > similar_minilm[1][1] # Score for e1 > score for e2
    assert isinstance(similar_minilm[0][2], MiniLMEmbeddingComponent) # Check component type

    # Query using CLIP (should only find entity3)
    # Query vec: [28, 16, 30, ...] (using clip model name for mock)
    # e3 (apple pie): [26, 9, 30, ...]
    similar_clip = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, model_name=TEST_MODEL_CLIP, model_params=TEST_PARAMS_CLIP, top_n=1
    )
    assert len(similar_clip) == 1
    assert similar_clip[0][0].id == entity3.id
    assert isinstance(similar_clip[0][2], ClipEmbeddingComponent)

    # Test with non-existent model in find_similar
    no_model_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, model_name="non-existent-model", model_params={}
    )
    assert len(no_model_results) == 0

    # Test with empty query text
    empty_query_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, "", model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    assert len(empty_query_results) == 0


@pytest.mark.asyncio
async def test_error_handling_in_update_embeddings(db_session: AsyncSession):
    entity = await ecs_service.create_entity(db_session)
    batch_for_error: List[BatchTextItem] = [
        {"component_name": "SourceE", "field_name": "field1", "text_content": "Error text 1"},
    ]

    # Simulate model.encode failing
    failing_mock_model_instance = MockSentenceTransformer(model_name_or_path=TEST_MODEL_MINILM)
    failing_mock_model_instance.encode = MagicMock(side_effect=RuntimeError("Simulated encoding error"))

    # Patch _load_model_sync to return this failing mock
    with patch("dam.services.semantic_service._load_model_sync", return_value=failing_mock_model_instance):
        semantic_service._model_cache.clear()
        error_comps = await semantic_service.update_text_embeddings_for_entity(
            db_session, entity.id, {}, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM, batch_texts=batch_for_error
        )
        assert len(error_comps) == 0
        db_error_comps = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
        assert len(db_error_comps) == 0
    semantic_service._model_cache.clear()

    # Simulate model.encode returning wrong number of embeddings
    wrong_count_mock_model_instance = MockSentenceTransformer(model_name_or_path=TEST_MODEL_MINILM)
    wrong_count_mock_model_instance.encode = MagicMock(return_value=np.array([])) # Empty array, not matching input len

    with patch("dam.services.semantic_service._load_model_sync", return_value=wrong_count_mock_model_instance):
        semantic_service._model_cache.clear()
        error_comps_wrong_count = await semantic_service.update_text_embeddings_for_entity(
            db_session, entity.id, {}, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM, batch_texts=batch_for_error
        )
        assert len(error_comps_wrong_count) == 0
        db_error_comps_wrong_count = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
        assert len(db_error_comps_wrong_count) == 0
    semantic_service._model_cache.clear()
