import pytest
from typer.testing import CliRunner
from unittest.mock import patch # For mocking service layer if needed for complex scenarios

from dam.cli import app # main Typer app
from dam.core.world import World
from dam.services import semantic_service, ecs_service # For setting up test data
from dam.models.properties import FilePropertiesComponent
from dam.models.semantic import TextEmbeddingComponent # To verify what the CLI might display

# Mock SentenceTransformer for CLI tests as well, to avoid real model loading
from .test_semantic_service import MockSentenceTransformer

runner = CliRunner()

@pytest.fixture(autouse=True)
def mock_sentence_transformer_for_cli_tests(monkeypatch):
    # Apply the same mock logic as in test_semantic_service
    monkeypatch.setattr('sentence_transformers.SentenceTransformer', MockSentenceTransformer)
    # Clear the service's model cache before each test
    if 'semantic_service' in globals(): # Ensure service is imported
        semantic_service._model_cache.clear()


@pytest.fixture(autouse=True)
async def current_test_world_for_search_cli(test_world_alpha: World):
    # This ensures 'test_world_alpha' is set up and used as the default for these CLI tests.
    yield test_world_alpha


@pytest.mark.asyncio
async def test_cli_search_semantic_no_results(current_test_world_for_search_cli: World):
    world_name = current_test_world_for_search_cli.name
    query = "non_existent_semantic_query"

    result = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
    print(f"CLI search semantic (no results) output: {result.stdout}")
    assert result.exit_code == 0
    assert f"No semantic matches found for query: '{query}" in result.stdout


@pytest.mark.asyncio
async def test_cli_search_semantic_with_results(current_test_world_for_search_cli: World):
    world = current_test_world_for_search_cli
    world_name = world.name

    # Setup: Create some entities and their embeddings
    async with world.db_session_maker() as session:
        entity1 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, entity1.id, FilePropertiesComponent(original_filename="apple_pie_doc.txt"))
        await semantic_service.update_text_embeddings_for_entity(
            session, entity1.id, {"Data.text": "apple pie"}, model_name=semantic_service.DEFAULT_MODEL_NAME
        )

        entity2 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, entity2.id, FilePropertiesComponent(original_filename="banana_bread_recipe.md"))
        await semantic_service.update_text_embeddings_for_entity(
            session, entity2.id, {"Data.text": "banana bread"}, model_name=semantic_service.DEFAULT_MODEL_NAME
        )
        await session.commit()

    query = "delicious apple pie recipe" # Mock will make this more similar to "apple pie"

    result = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "1"])
    print(f"CLI search semantic (with results) output: {result.stdout}")
    assert result.exit_code == 0
    assert "Semantic Search Results" in result.stdout
    assert f"Found 1 results for query '{query}" in result.stdout # Query might be truncated in output
    assert f"Entity ID: {entity1.id}" in result.stdout
    assert "apple_pie_doc.txt" in result.stdout
    assert "Data.text" in result.stdout # Source of matched embedding
    assert semantic_service.DEFAULT_MODEL_NAME in result.stdout

    # Ensure entity2 is not in top 1 (or has lower score if top-n was higher)
    assert f"Entity ID: {entity2.id}" not in result.stdout

    # Test with higher top_n to get both
    result_top2 = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "2"])
    assert result_top2.exit_code == 0
    assert f"Found 2 results" in result_top2.stdout
    assert f"Entity ID: {entity1.id}" in result_top2.stdout
    assert f"Entity ID: {entity2.id}" in result_top2.stdout # banana bread should appear with lower score

    # Verify order (apple pie should be first due to mock similarity)
    output_lines = result_top2.stdout.splitlines()
    entity1_line_index = -1
    entity2_line_index = -1
    for i, line in enumerate(output_lines):
        if f"Entity ID: {entity1.id}" in line:
            entity1_line_index = i
        if f"Entity ID: {entity2.id}" in line:
            entity2_line_index = i

    assert entity1_line_index != -1 and entity2_line_index != -1
    assert entity1_line_index < entity2_line_index # entity1 (apple pie) should appear before entity2 (banana bread)


@pytest.mark.asyncio
async def test_cli_search_semantic_model_loading_error(current_test_world_for_search_cli: World):
    world_name = current_test_world_for_search_cli.name
    query = "test query"

    # Mock get_sentence_transformer_model to raise an error
    with patch('dam.services.semantic_service.get_sentence_transformer_model', side_effect=Exception("Mock Model Load Fail")):
        result = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
        print(f"CLI search semantic (model error) output: {result.stdout}")
        assert result.exit_code != 0 # Should indicate failure
        assert "Semantic search query failed" in result.stdout
        assert "Mock Model Load Fail" in result.stdout


@pytest.mark.asyncio
async def test_cli_search_items_placeholder(current_test_world_for_search_cli: World):
    world_name = current_test_world_for_search_cli.name
    result = runner.invoke(app, ["--world", world_name, "search", "items", "--text", "test"])
    assert result.exit_code == 0 # Placeholder command doesn't exit with error
    assert "Item search CLI is a work in progress." in result.stdout

# TODO: Add tests for --model option in semantic search CLI once service supports it more explicitly
# or if we want to test passing it through.
# For now, the mock uses a fixed model name logic or default.
