import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from dam.services import semantic_service, ecs_service
from dam.models.semantic import TextEmbeddingComponent
from dam.models.core.entity import Entity

# Mock SentenceTransformer class
class MockSentenceTransformer:
    def __init__(self, model_name_or_path=None):
        self.model_name = model_name_or_path
        # Simple mock: return a fixed vector based on text length or a hash
        # For more controlled tests, you might want to set up specific text -> vector mappings.

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for s in sentences:
            if not s or not s.strip(): # Handle empty strings like the service does
                 # This case should ideally be filtered out before calling encode by the service logic.
                 # If it reaches here, the behavior depends on the actual SentenceTransformer.
                 # For mock, let's return a zero vector or raise error if that's more appropriate.
                 # The service's generate_embedding filters empty strings, so this path might not be hit often.
                embeddings.append(np.zeros(384, dtype=np.float32)) # Assuming 384 dim like 'all-MiniLM-L6-v2'
                continue

            # Create a simple deterministic "embedding"
            sum_ords = sum(ord(c) for c in s)
            vec = np.array([sum_ords % 100, len(s) % 100] + [0.0] * 382, dtype=np.float32)
            embeddings.append(vec)

        if not convert_to_numpy: # Though our service uses convert_to_numpy=True
            return [e.tolist() for e in embeddings]
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0] if embeddings else np.array([])


@pytest.fixture(autouse=True)
def mock_sentence_transformer_loader():
    # This fixture will automatically apply the patch for all tests in this module.
    with patch('sentence_transformers.SentenceTransformer', new=MockSentenceTransformer) as mock_st:
        # Clear the service's model cache before each test to ensure fresh mocks are used if model names change
        semantic_service._model_cache.clear()
        yield mock_st


@pytest.mark.asyncio
async def test_generate_embedding_and_conversion():
    text = "Hello world"
    embedding_np = semantic_service.generate_embedding(text, model_name="mock-model")
    assert embedding_np is not None
    assert isinstance(embedding_np, np.ndarray)
    assert embedding_np.shape == (384,) # Mocked shape, based on all-MiniLM-L6-v2

    # Test empty text
    assert semantic_service.generate_embedding("", model_name="mock-model") is None
    assert semantic_service.generate_embedding("   ", model_name="mock-model") is None

    # Test conversion
    embedding_bytes = semantic_service.convert_embedding_to_bytes(embedding_np)
    assert isinstance(embedding_bytes, bytes)
    # Expected length for 384 float32 values: 384 * 4 = 1536
    assert len(embedding_bytes) == 1536

    embedding_np_restored = semantic_service.convert_bytes_to_embedding(embedding_bytes)
    assert np.array_equal(embedding_np, embedding_np_restored)


@pytest.mark.asyncio
async def test_update_text_embeddings_for_entity(db_session: AsyncSession):
    entity = await ecs_service.create_entity(db_session)

    text_map = {
        "FilePropertiesComponent.original_filename": "My Test Document.pdf",
        "DescriptionComponent.text": "This is a sample document about testing."
    }
    model_name = "test-model-1"

    # First time: create embeddings
    created_components = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map, model_name=model_name
    )
    assert len(created_components) == 2

    db_emb_comps = await ecs_service.get_components(db_session, entity.id, TextEmbeddingComponent)
    assert len(db_emb_comps) == 2

    found_sources = set()
    for comp in db_emb_comps:
        assert comp.model_name == model_name
        assert comp.embedding_vector is not None
        source_key = f"{comp.source_component_name}.{comp.source_field_name}"
        found_sources.add(source_key)
        # Verify embedding content (optional, depends on mock predictability)
        original_text = text_map[source_key]
        expected_mock_emb = MockSentenceTransformer(model_name).encode(original_text)
        assert np.array_equal(semantic_service.convert_bytes_to_embedding(comp.embedding_vector), expected_mock_emb)

    assert "FilePropertiesComponent.original_filename" in found_sources
    assert "DescriptionComponent.text" in found_sources

    # Second time: update embeddings (if text changes - mock won't change unless text does)
    # Let's change one text
    text_map_updated = {
        "FilePropertiesComponent.original_filename": "My Test Document Updated.pdf", # Changed
        "DescriptionComponent.text": "This is a sample document about testing." # Unchanged
    }
    updated_components = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, text_map_updated, model_name=model_name
    )
    assert len(updated_components) == 2 # Both processed, one updated, one identified as same

    db_emb_comps_after_update = await ecs_service.get_components(db_session, entity.id, TextEmbeddingComponent)
    assert len(db_emb_comps_after_update) == 2 # Still 2 components

    for comp in db_emb_comps_after_update:
        source_key = f"{comp.source_component_name}.{comp.source_field_name}"
        original_text = text_map_updated[source_key]
        expected_mock_emb = MockSentenceTransformer(model_name).encode(original_text)
        actual_emb = semantic_service.convert_bytes_to_embedding(comp.embedding_vector)
        assert np.array_equal(actual_emb, expected_mock_emb)
        if source_key == "FilePropertiesComponent.original_filename":
            # This one should have a different vector than the very first one
            original_first_text = text_map["FilePropertiesComponent.original_filename"]
            original_first_emb = MockSentenceTransformer(model_name).encode(original_first_text)
            assert not np.array_equal(actual_emb, original_first_emb)

    # Test with batch_texts
    await ecs_service.delete_entity(db_session, entity.id) # Clean up previous entity
    entity2 = await ecs_service.create_entity(db_session)
    batch = [
        ("SourceA", "field1", "Text for A1"),
        ("SourceA", "field2", "Text for A2"),
    ]
    batched_comps = await semantic_service.update_text_embeddings_for_entity(
        db_session, entity2.id, {}, model_name=model_name, batch_texts=batch
    )
    assert len(batched_comps) == 2
    db_batch_comps = await ecs_service.get_components(db_session, entity2.id, TextEmbeddingComponent)
    assert len(db_batch_comps) == 2


@pytest.mark.asyncio
async def test_get_text_embeddings_for_entity(db_session: AsyncSession):
    entity = await ecs_service.create_entity(db_session)
    text_map = {
        "CompA.field1": "Text for model A",
        "CompB.field2": "Text for model B"
    }
    await semantic_service.update_text_embeddings_for_entity(db_session, entity.id, text_map, model_name="model-A")
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity.id, {"CompC.field3": "Text for model C"}, model_name="model-C"
    )

    all_embs = await semantic_service.get_text_embeddings_for_entity(db_session, entity.id)
    assert len(all_embs) == 3 # 2 from model-A, 1 from model-C

    model_a_embs = await semantic_service.get_text_embeddings_for_entity(db_session, entity.id, model_name="model-A")
    assert len(model_a_embs) == 2
    assert all(e.model_name == "model-A" for e in model_a_embs)

    model_c_embs = await semantic_service.get_text_embeddings_for_entity(db_session, entity.id, model_name="model-C")
    assert len(model_c_embs) == 1
    assert model_c_embs[0].model_name == "model-C"
    assert model_c_embs[0].source_component_name == "CompC"


@pytest.mark.asyncio
async def test_find_similar_entities_by_text_embedding(db_session: AsyncSession):
    entity1 = await ecs_service.create_entity(db_session)
    entity2 = await ecs_service.create_entity(db_session)
    entity3 = await ecs_service.create_entity(db_session)

    # Populate embeddings - using simple texts where similarity can be somewhat controlled by the mock
    # Mock embedding: np.array([sum(ord(c)) % 100, len(s) % 100] + [0.0] * 382
    # Text1: "apple pie" -> sum_ords=826, len=9. vec1 = [26, 9, 0...]
    # Text2: "apple crumble" -> sum_ords=1230, len=13. vec2 = [30, 13, 0...] (more similar to text1 if sum_ords is closer)
    # Text3: "banana bread" -> sum_ords=1100, len=12. vec3 = [0, 12, 0...] (less similar)
    # Text4: "apple pie recipe" -> sum_ords=1528, len=16. vec4 = [28, 0, 0...] (query text)

    model = "similarity-test-model"
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity1.id, {"Data.text": "apple pie"}, model_name=model
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity2.id, {"Data.text": "apple crumble"}, model_name=model
    )
    await semantic_service.update_text_embeddings_for_entity(
        db_session, entity3.id, {"Data.text": "banana bread"}, model_name=model
    )

    query_text = "apple pie recipe" # vec_query = [28,0,...]

    # Vec1 (apple pie): [26, 9, ...]
    # Vec2 (apple crumble): [30, 13, ...]
    # Vec3 (banana bread): [0, 12, ...]
    # Query (apple pie recipe): [28, 0, ...]

    # Cosine similarity: (A dot B) / (||A|| * ||B||)
    # A=[28,0], B1=[26,9] -> dot=28*26=728. ||A||=28. ||B1||=sqrt(26^2+9^2)=sqrt(676+81)=sqrt(757)=27.5
    # sim1 = 728 / (28 * 27.5) = 728 / 770 = 0.945

    # A=[28,0], B2=[30,13] -> dot=28*30=840. ||A||=28. ||B2||=sqrt(30^2+13^2)=sqrt(900+169)=sqrt(1069)=32.69
    # sim2 = 840 / (28*32.69) = 840 / 915.32 = 0.917

    # A=[28,0], B3=[0,12] -> dot=0. ||A||=28. ||B3||=12
    # sim3 = 0

    # Expected order: entity1, entity2, entity3

    similar_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, top_n=3, model_name=model
    )

    assert len(similar_results) == 3
    # Results are (Entity, score, TextEmbeddingComponent)
    # Check order by score (descending) then by entity ID if scores are identical (not expected here)

    assert similar_results[0][0].id == entity1.id # apple pie (score ~0.945)
    assert similar_results[1][0].id == entity2.id # apple crumble (score ~0.917)
    assert similar_results[2][0].id == entity3.id # banana bread (score 0)

    assert similar_results[0][1] > similar_results[1][1]
    assert similar_results[1][1] > similar_results[2][1]
    assert pytest.approx(similar_results[2][1]) == 0.0

    # Test with top_n
    top_1_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, top_n=1, model_name=model
    )
    assert len(top_1_results) == 1
    assert top_1_results[0][0].id == entity1.id

    # Test with non-existent model (should return empty or handle gracefully)
    no_model_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, model_name="non-existent-model"
    )
    assert len(no_model_results) == 0

    # Test with empty query text
    empty_query_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, "", model_name=model
    )
    assert len(empty_query_results) == 0

    # Test when no embeddings exist in DB for the model
    await ecs_service.delete_entity(db_session, entity1.id) # This also deletes its components
    await ecs_service.delete_entity(db_session, entity2.id)
    await ecs_service.delete_entity(db_session, entity3.id)

    no_db_embeddings_results = await semantic_service.find_similar_entities_by_text_embedding(
        db_session, query_text, model_name=model
    )
    assert len(no_db_embeddings_results) == 0
