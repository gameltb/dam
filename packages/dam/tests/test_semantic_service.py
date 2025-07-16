from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ModelExecutionManager fixture from conftest.py
from dam.core.model_manager import ModelExecutionManager

# Models to be tested (actual specific component classes)
from dam.models.semantic import (
    ModelHyperparameters,
    get_embedding_component_class,
)
from dam.services import ecs_service, semantic_service
from dam.services.semantic_service import BatchTextItem  # Keep this for type hints
from sqlalchemy.ext.asyncio import AsyncSession

from .conftest import MockSentenceTransformer  # Import the mock from conftest

# Define model names and params to be used in tests, corresponding to registered models
TEST_MODEL_MINILM = "all-MiniLM-L6-v2"
TEST_PARAMS_MINILM: ModelHyperparameters = {"dimensions": 384}  # Must match registry

TEST_MODEL_CLIP = "clip-ViT-B-32"
TEST_PARAMS_CLIP: ModelHyperparameters = {"dimensions": 512}  # Must match registry

# Get the component classes for easier use in tests
MiniLMEmbeddingComponent = get_embedding_component_class(TEST_MODEL_MINILM, TEST_PARAMS_MINILM)
ClipEmbeddingComponent = get_embedding_component_class(TEST_MODEL_CLIP, TEST_PARAMS_CLIP)

# Ensure the test models are actually registered and component classes are found
assert MiniLMEmbeddingComponent is not None, "Test MiniLM model not registered or class not found"
assert ClipEmbeddingComponent is not None, "Test CLIP model not registered or class not found"

# MockSentenceTransformer class is now defined in conftest.py
# The global_mock_sentence_transformer_loader fixture in conftest.py also handles
# patching _load_model_sync and clearing the cache.


@pytest.mark.asyncio
async def test_generate_embedding_and_conversion(
    test_world_alpha: Any,
    global_model_execution_manager: ModelExecutionManager,  # Use fixture
):
    # test_world_alpha fixture ensures the world "test_world_alpha" is created and registered.
    text = "Hello world"
    # Test with MiniLM
    embedding_minilm_np = await semantic_service.generate_embedding(
        global_model_execution_manager,  # Pass MEM
        text,
        model_name=TEST_MODEL_MINILM,
        params=TEST_PARAMS_MINILM,
        # world_name="test_world_alpha" # Removed
    )
    assert embedding_minilm_np is not None
    assert isinstance(embedding_minilm_np, np.ndarray)
    assert embedding_minilm_np.shape == (TEST_PARAMS_MINILM["dimensions"],)

    # Test with CLIP
    embedding_clip_np = await semantic_service.generate_embedding(
        global_model_execution_manager,  # Pass MEM
        text,
        model_name=TEST_MODEL_CLIP,
        params=TEST_PARAMS_CLIP,
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
    # Empty text check is done before model loading, so world_name might not be strictly needed if it bails early.
    assert (
        await semantic_service.generate_embedding(
            global_model_execution_manager, "", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )
    assert (
        await semantic_service.generate_embedding(
            global_model_execution_manager, "   ", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )

    # Test conversion (using MiniLM embedding)
    embedding_bytes = semantic_service.convert_embedding_to_bytes(embedding_minilm_np)
    assert isinstance(embedding_bytes, bytes)
    assert len(embedding_bytes) == TEST_PARAMS_MINILM["dimensions"] * 4  # 4 bytes per float32

    embedding_np_restored = semantic_service.convert_bytes_to_embedding(embedding_bytes)
    assert np.array_equal(embedding_minilm_np, embedding_np_restored)

    # Test with unregistered model name (should fail to get component class)
    # generate_embedding itself doesn't directly use get_embedding_component_class,
    # it uses get_sentence_transformer_model. So this call might still "succeed" at generating
    # if the mock loader can handle "unregistered-model".
    # The failure would occur in update_text_embeddings_for_entity.
    # However, get_sentence_transformer_model now uses registry for default params, so it might log warning.
    unregistered_embedding = await semantic_service.generate_embedding(
        global_model_execution_manager, "test", model_name="unregistered-model", params={}
    )
    # Depending on mock and error handling, this might return an embedding or raise.
    # MockSentenceTransformer will be created.
    assert unregistered_embedding is not None


@pytest.mark.asyncio
async def test_update_text_embeddings_for_entity_specific_tables(
    db_session: AsyncSession,
    global_model_execution_manager: ModelExecutionManager,  # Use fixture
):
    entity = await ecs_service.create_entity(db_session)
    text_map = {
        "File.name": "document.pdf",
        "Desc.text": "Sample content.",
    }

    # Update with MiniLM model
    created_minilm = await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        text_map,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
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
            global_model_execution_manager,  # Pass MEM
            original_text,
            model_name=TEST_MODEL_MINILM,
            params=TEST_PARAMS_MINILM,
        )
        assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb)

    # Update with CLIP model (should go to a different table)
    created_clip = await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        text_map,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_CLIP,
        model_params=TEST_PARAMS_CLIP,
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
            global_model_execution_manager,  # Pass MEM
            original_text,
            model_name=TEST_MODEL_CLIP,
            params=TEST_PARAMS_CLIP,
        )
        assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb)

    # Ensure MiniLM table still has its original 2 components
    db_minilm_comps_after_clip = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
    assert len(db_minilm_comps_after_clip) == 2

    # Test update (text change) for one model
    text_map_updated = {"File.name": "new_document.pdf", "Desc.text": "Sample content."}
    updated_minilm = await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        text_map_updated,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
    )
    assert len(updated_minilm) == 2
    db_minilm_updated = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
    assert len(db_minilm_updated) == 2  # Still 2 components
    for comp in db_minilm_updated:
        if comp.source_component_name == "File":
            expected_emb_updated = await semantic_service.generate_embedding(
                global_model_execution_manager,  # Pass MEM
                "new_document.pdf",
                model_name=TEST_MODEL_MINILM,
                params=TEST_PARAMS_MINILM,
            )
            assert np.array_equal(
                semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_emb_updated
            )
            original_emb_old = await semantic_service.generate_embedding(
                global_model_execution_manager,  # Pass MEM
                "document.pdf",
                model_name=TEST_MODEL_MINILM,
                params=TEST_PARAMS_MINILM,
            )
            assert not np.array_equal(
                semantic_service.convert_bytes_to_embedding(comp.embedding_vector), original_emb_old
            )

    # Test with an unregistered model - should return empty list and log error
    unregistered_results = await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        text_map,
        global_model_execution_manager,
        model_name="unregistered-model",
        model_params={},
    )
    assert len(unregistered_results) == 0


@pytest.mark.asyncio
async def test_get_text_embeddings_for_entity(
    db_session: AsyncSession,
    global_model_execution_manager: ModelExecutionManager,  # Use fixture
):
    entity = await ecs_service.create_entity(db_session)
    text_map = {"CompA.field1": "Text A", "CompB.field2": "Text B"}

    await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        text_map,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity.id,
        {"CompC.field3": "Text C"},
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_CLIP,
        model_params=TEST_PARAMS_CLIP,
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
async def test_find_similar_entities_by_text_embedding(
    db_session: AsyncSession,
    global_model_execution_manager: ModelExecutionManager,  # Use fixture
):
    entity1 = await ecs_service.create_entity(db_session)  # Target for MiniLM
    entity2 = await ecs_service.create_entity(db_session)  # Target for MiniLM
    entity3 = await ecs_service.create_entity(db_session)  # Target for CLIP

    # Populate with MiniLM embeddings
    # Mock embedding: np.array([sum_ords % 100, len(s) % 100, model_ord_sum % 100] + [0.0] * (dim - 3))
    # model_ord_sum for "all-MiniLM-L6-v2" = 1841 % 100 = 41
    # Text1: "apple pie" -> sum_ords=826, len=9. vec1_minilm = [26, 9, 41, 0...]
    # Text2: "apple crumble" -> sum_ords=1230, len=13. vec2_minilm = [30, 13, 41, 0...]
    await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity1.id,
        {"Data.text": "apple pie"},
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity2.id,
        {"Data.text": "apple crumble"},
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
    )

    # Populate with CLIP embeddings (should not interfere with MiniLM search) (MEM passed to update_text_embeddings_for_entity)
    # model_ord_sum for "clip-ViT-B-32" = 1230 % 100 = 30
    # Text3: "apple pie" (same text, different model) -> vec3_clip = [26, 9, 30, 0...]
    await semantic_service.update_text_embeddings_for_entity(
        db_session,
        entity3.id,
        {"Data.text": "apple pie"},
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_CLIP,
        model_params=TEST_PARAMS_CLIP,
    )

    # Query using MiniLM (MEM passed to find_similar_entities_by_text_embedding)
    query_text = "apple pie recipe"  # sum_ords=1528, len=16. vec_query_minilm = [28, 16, 41, 0...]
    # Using mock: sum_ords=1528 -> 28, len=16 -> 16.
    # model_ord_sum for "all-MiniLM-L6-v2" is 41.
    # Query vec: [28, 16, 41, ...]
    # e1 (apple pie): [26, 9, 41, ...]
    # e2 (apple crumble): [30, 13, 41, ...]
    # Cosine sim: query vs e1 should be higher than query vs e2.

    similar_minilm = await semantic_service.find_similar_entities_by_text_embedding(
        db_session,
        query_text,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_MINILM,
        model_params=TEST_PARAMS_MINILM,
        top_n=2,
    )
    assert len(similar_minilm) == 2
    assert similar_minilm[0][0].id == entity1.id  # apple pie
    assert similar_minilm[1][0].id == entity2.id  # apple crumble
    assert similar_minilm[0][1] > similar_minilm[1][1]  # Score for e1 > score for e2
    assert isinstance(similar_minilm[0][2], MiniLMEmbeddingComponent)  # Check component type

    # Query using CLIP (should only find entity3)
    # Query vec: [28, 16, 30, ...] (using clip model name for mock)
    # e3 (apple pie): [26, 9, 30, ...]
    similar_clip = await semantic_service.find_similar_entities_by_text_embedding(
        db_session,
        query_text,
        global_model_execution_manager,  # Pass MEM
        model_name=TEST_MODEL_CLIP,
        model_params=TEST_PARAMS_CLIP,
        top_n=1,
    )
    assert len(similar_clip) == 1
    assert similar_clip[0][0].id == entity3.id
    assert isinstance(similar_clip[0][2], ClipEmbeddingComponent)

    # Test with non-existent model in find_similar
    no_model_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, global_model_execution_manager, model_name="non-existent-model", model_params={}
    )
    assert len(no_model_results) == 0

    # Test with empty query text
    empty_query_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, "", global_model_execution_manager, model_name=TEST_MODEL_MINILM, model_params=TEST_PARAMS_MINILM
    )
    assert len(empty_query_results) == 0


@pytest.mark.asyncio
async def test_error_handling_in_update_embeddings(
    db_session: AsyncSession,
    global_model_execution_manager: ModelExecutionManager,  # Use fixture
):
    entity = await ecs_service.create_entity(db_session)
    batch_for_error: List[BatchTextItem] = [
        {"component_name": "SourceE", "field_name": "field1", "text_content": "Error text 1"},
    ]

    # Simulate model.encode failing
    failing_mock_model_instance = MockSentenceTransformer(model_name_or_path=TEST_MODEL_MINILM)
    failing_mock_model_instance.encode = MagicMock(side_effect=RuntimeError("Simulated encoding error"))

    # Patch _load_sentence_transformer_model_sync to return this failing mock
    # This test is tricky because get_sentence_transformer_model is called inside update_text_embeddings_for_entity
    # and it uses the ModelExecutionManager from the world.
    # The global_mock_sentence_transformer_loader in conftest.py might interfere or make this specific patch target difficult.
    # For this test to work as intended (testing error handling within update_text_embeddings_for_entity due to loader issues),
    # we need to ensure our local patch here takes precedence or that the global mock doesn't prevent the side_effect.

    # The global mock in conftest.py patches `_load_sentence_transformer_model_sync` directly.
    # So, to test a failure *of that loading function*, we need to make the *mocked version* (MockSentenceTransformer's creation) fail,
    # or mock the `ModelExecutionManager.get_model` call if that's easier.

    # Let's assume the global mock is active. We want `get_sentence_transformer_model` to raise an error.
    # We can achieve this by patching `get_sentence_transformer_model` itself for this specific test.

    with patch(
        "dam.services.semantic_service.get_sentence_transformer_model",
        side_effect=RuntimeError("Simulated model loading error"),
    ):
        error_comps = await semantic_service.update_text_embeddings_for_entity(
            db_session,
            entity.id,
            {},
            global_model_execution_manager,  # This call will fail due to the patch above
            batch_texts=batch_for_error,
            model_name=TEST_MODEL_MINILM,
            model_params=TEST_PARAMS_MINILM,  # world_name removed
        )
        assert len(error_comps) == 0  # Expect no components created due to error
        db_error_comps = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
        assert len(db_error_comps) == 0  # No components should be in DB

    # Test when model.encode itself fails (after successful loading)
    # The global mock loader in conftest.py returns MockSentenceTransformer.
    # We need to make its `encode` method fail.
    # This requires getting the mock instance that will be returned by the loader.
    # The global mock is already in place. We need to make *that* mock's encode fail.
    # This is hard to do here without more complex mock setup for the global mock.

    # Alternative: Patch `MockSentenceTransformer.encode` if the global mock is predictable.
    # This is fragile. A better way:
    # Temporarily unpatch the global mock, apply a local failing mock loader, then restore.
    # Or, more simply, make `get_sentence_transformer_model` return our own failing_mock_model_instance.

    with patch(
        "dam.services.semantic_service.get_sentence_transformer_model", return_value=failing_mock_model_instance
    ):
        error_comps_encode_fail = await semantic_service.update_text_embeddings_for_entity(
            db_session,
            entity.id,
            {},
            global_model_execution_manager,  # This will be used by the patched get_sentence_transformer_model
            batch_texts=batch_for_error,
            model_name=TEST_MODEL_MINILM,
            model_params=TEST_PARAMS_MINILM,  # world_name removed
        )
        assert len(error_comps_encode_fail) == 0
        db_error_comps_encode_fail = await ecs_service.get_components(db_session, entity.id, MiniLMEmbeddingComponent)
        assert len(db_error_comps_encode_fail) == 0
