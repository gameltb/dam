import pytest
from typer.testing import CliRunner
from unittest.mock import patch # For mocking service layer if needed for complex scenarios

from dam.cli import app # main Typer app
from dam.core.world import World
from dam.services import semantic_service, ecs_service # For setting up test data
from dam.models.properties import FilePropertiesComponent
from dam.models.semantic import TextEmbeddingComponent # To verify what the CLI might display

# MockSentenceTransformer is now globally available from conftest.py
# from .test_semantic_service import MockSentenceTransformer # Removed

# runner = CliRunner() # Removed global runner

# Removed local mock_sentence_transformer_for_cli_tests, now handled by conftest.py
# @pytest.fixture(autouse=True)
# def mock_sentence_transformer_for_cli_tests(monkeypatch):
#     # Apply the same mock logic as in test_semantic_service
#     monkeypatch.setattr('sentence_transformers.SentenceTransformer', MockSentenceTransformer)
#     # Clear the service's model cache before each test
#     if 'semantic_service' in globals(): # Ensure service is imported
#         semantic_service._model_cache.clear()


@pytest.fixture(autouse=True)
async def current_test_world_for_search_cli(test_world_alpha: World):
    # This ensures 'test_world_alpha' is set up and used as the default for these CLI tests.
    yield test_world_alpha


import asyncio # Add asyncio import if not already present at top

# @pytest.mark.asyncio # Removed
def test_cli_search_semantic_no_results(current_test_world_for_search_cli: World, click_runner: CliRunner):
    world_name = current_test_world_for_search_cli.name
    query = "non_existent_semantic_query"

    result = click_runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
    print(f"CLI search semantic (no results) output: {result.stdout}")
    assert result.exit_code == 0
    assert f"No semantic matches found for query: '{query}" in result.stdout


# @pytest.mark.asyncio # Removed
def test_cli_search_semantic_with_results(current_test_world_for_search_cli: World, click_runner: CliRunner):
    world = current_test_world_for_search_cli
    world_name = world.name

    # Variables to store entity IDs from async setup
    entity1_id: Optional[int] = None
    entity2_id: Optional[int] = None

    async def setup_data():
        nonlocal entity1_id, entity2_id
        async with world.db_session_maker() as session:
            entity1 = await ecs_service.create_entity(session)
            entity1_id = entity1.id
            await ecs_service.add_component_to_entity(session, entity1.id, FilePropertiesComponent(original_filename="apple_pie_doc.txt", file_size_bytes=100, mime_type="text/plain"))
            await semantic_service.update_text_embeddings_for_entity(
                session, entity1.id, {"Data.text": "apple pie"}, model_name=semantic_service.DEFAULT_MODEL_NAME
            )

            entity2 = await ecs_service.create_entity(session)
            entity2_id = entity2.id
            await ecs_service.add_component_to_entity(session, entity2.id, FilePropertiesComponent(original_filename="banana_bread_recipe.md", file_size_bytes=100, mime_type="text/markdown"))
            await semantic_service.update_text_embeddings_for_entity(
                session, entity2.id, {"Data.text": "banana bread"}, model_name=semantic_service.DEFAULT_MODEL_NAME
            )
            await session.commit()

    asyncio.run(setup_data())
    assert entity1_id is not None and entity2_id is not None

    query = "delicious apple pie recipe"

    result = click_runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "1"])
    print(f"CLI search semantic (with results) output: {result.stdout}")
    assert result.exit_code == 0
    assert "Semantic Search Results" in result.stdout
    assert f"Found 1 results for query '{query}" in result.stdout
    assert f"Entity ID: {entity1_id}" in result.stdout
    assert "apple_pie_doc.txt" in result.stdout
    assert "Data.text" in result.stdout
    assert semantic_service.DEFAULT_MODEL_NAME in result.stdout


    assert f"Entity ID: {entity2_id}" not in result.stdout


    result_top2 = click_runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "2"])
    assert result_top2.exit_code == 0
    assert f"Found 2 results" in result_top2.stdout
    assert f"Entity ID: {entity1_id}" in result_top2.stdout
    assert f"Entity ID: {entity2_id}" in result_top2.stdout


    output_lines = result_top2.stdout.splitlines()
    entity1_line_index = -1
    entity2_line_index = -1
    for i, line in enumerate(output_lines):
        if f"Entity ID: {entity1_id}" in line:
            entity1_line_index = i
        if f"Entity ID: {entity2_id}" in line:
            entity2_line_index = i

    assert entity1_line_index != -1 and entity2_line_index != -1
    assert entity1_line_index < entity2_line_index


# @pytest.mark.asyncio # Removed
def test_cli_search_semantic_model_loading_error(current_test_world_for_search_cli: World, click_runner: CliRunner):
    world_name = current_test_world_for_search_cli.name
    query = "test query"



    with patch('dam.services.semantic_service.find_similar_entities_by_text_embedding', side_effect=Exception("Mock Search Service Fail")):
        result = click_runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
        print(f"CLI search semantic (model error) output: {result.stdout}")
        assert result.exit_code != 0
        assert "Semantic search query failed" in result.stdout
        assert "Mock Search Service Fail" in result.stdout


# @pytest.mark.asyncio # Removed
def test_cli_search_items_placeholder(current_test_world_for_search_cli: World, click_runner: CliRunner):
    world_name = current_test_world_for_search_cli.name
    result = click_runner.invoke(app, ["--world", world_name, "search", "items", "--text", "test"])
    assert result.exit_code == 0
    assert "Item search CLI is a work in progress." in result.stdout

# TODO: Add tests for --model option in semantic search CLI once service supports it more explicitly
# or if we want to test passing it through.
# For now, the mock uses a fixed model name logic or default.
