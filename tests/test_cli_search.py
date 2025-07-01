# @pytest.mark.asyncio # Removed
def test_cli_search_semantic_with_results(current_test_world_for_search_cli: World, click_runner: CliRunner):
    world = current_test_world_for_search_cli
    world_name = world.name

    # Variables to store entity IDs from async setup
    entity1_id: Optional[int] = None # apple pie
    entity2_id: Optional[int] = None # banana bread

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
    # Based on re-calculation, entity2 (banana bread) should be more similar with the mock
    assert f"Entity ID: {entity2_id}" in result.stdout
    assert "banana_bread_recipe.md" in result.stdout
    assert "Data.text" in result.stdout
    assert semantic_service.DEFAULT_MODEL_NAME in result.stdout


    assert f"Entity ID: {entity1_id}" not in result.stdout


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
    # entity2 (banana bread) should be more similar and appear before entity1 (apple pie)
    assert entity2_line_index < entity1_line_index
