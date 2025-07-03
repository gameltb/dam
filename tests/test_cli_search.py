from unittest.mock import patch
import pytest
from typer.testing import CliRunner
import asyncio # Ensure asyncio is imported
from typing import Optional # Ensure Optional is imported

from dam.cli import app
from dam.core.world import World
from dam.models.properties import FilePropertiesComponent
from dam.services import ecs_service, semantic_service
from .conftest import MockSentenceTransformer

runner = CliRunner()

@pytest.fixture(autouse=True)
async def current_test_world_for_search_cli(test_world_alpha: World):
    yield test_world_alpha

def test_cli_search_semantic_no_results(current_test_world_for_search_cli: World):
    world_name = current_test_world_for_search_cli.name
    query = "non_existent_semantic_query"
    result = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
    print(f"CLI search semantic (no results) output: {result.stdout}")
    assert result.exit_code == 0
    assert f"No semantic matches found for query: '{query}" in result.stdout

async def test_cli_search_semantic_with_results(current_test_world_for_search_cli: World, click_runner: CliRunner): # Made async
    world = current_test_world_for_search_cli
    world_name = world.name

    entity1_id: Optional[int] = None
    entity2_id: Optional[int] = None

    async def setup_data():
        nonlocal entity1_id, entity2_id
        async with world.db_session_maker() as session:
            entity1 = await ecs_service.create_entity(session)
            entity1_id = entity1.id
            await ecs_service.add_component_to_entity(
                session,
                entity1.id,
                FilePropertiesComponent(
                    original_filename="apple_pie_doc.txt", file_size_bytes=100, mime_type="text/plain"
                ),
            )
            # Pass world_name here
            await semantic_service.update_text_embeddings_for_entity(
                session, entity1.id, {"Data.text": "apple pie"},
                model_name=semantic_service.DEFAULT_MODEL_NAME,
                world_name=world.name
            )

            entity2 = await ecs_service.create_entity(session)
            entity2_id = entity2.id
            await ecs_service.add_component_to_entity(
                session,
                entity2.id,
                FilePropertiesComponent(
                    original_filename="banana_bread_recipe.md", file_size_bytes=100, mime_type="text/markdown"
                ),
            )
            # world_name is already correctly passed here in the provided snippet
            await semantic_service.update_text_embeddings_for_entity(
                    session, entity2.id, {"Data.text": "banana bread"},
                    model_name=semantic_service.DEFAULT_MODEL_NAME,
                    world_name=world.name
            )
            await session.commit()

    await setup_data() # Changed from asyncio.run()
    assert entity1_id is not None and entity2_id is not None

    query = "delicious apple pie recipe"

    result = click_runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "1"])
    print(f"CLI search semantic (with results) output: {result.stdout}")
    assert result.exit_code == 0
    assert "Semantic Search Results" in result.stdout
    assert f"Found 1 results for query '{query}" in result.stdout
    # This assertion needs to be re-evaluated based on actual mock behavior after fix
    # For "delicious apple pie recipe", "apple pie" (entity1) should be more similar
    assert f"Entity ID: {entity1_id}" in result.stdout
    assert "apple_pie_doc.txt" in result.stdout
    assert "Data.text" in result.stdout # Source field of the embedding
    assert semantic_service.DEFAULT_MODEL_NAME in result.stdout # Model name used

    assert f"Entity ID: {entity2_id}" not in result.stdout # banana bread should not be the top 1

    result_top2 = click_runner.invoke(
        app, ["--world", world_name, "search", "semantic", "--query", query, "--top-n", "2"]
    )
    assert result_top2.exit_code == 0
    assert "Found 2 results" in result_top2.stdout
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
    # apple pie (entity1) should be more similar and appear before banana bread (entity2)
    assert entity1_line_index < entity2_line_index


def test_cli_search_semantic_model_loading_error(
    current_test_world_for_search_cli: World, click_runner: CliRunner
):
    world_name = current_test_world_for_search_cli.name
    query = "test query"
    with patch(
        "dam.services.semantic_service.find_similar_entities_by_text_embedding",
        side_effect=Exception("Mock Search Service Fail"),
    ):
        result = runner.invoke(app, ["--world", world_name, "search", "semantic", "--query", query])
        print(f"CLI search semantic (model error) output: {result.stdout}")
        assert result.exit_code != 0
        assert "Semantic search query failed" in result.stdout
        assert "Mock Search Service Fail" in result.stdout

def test_cli_search_items_placeholder(current_test_world_for_search_cli: World):
    world_name = current_test_world_for_search_cli.name
    result = runner.invoke(app, ["--world", world_name, "search", "items", "--text", "test"])
    assert result.exit_code == 0
    assert "Item search CLI is a work in progress." in result.stdout
