import pytest
from typer.testing import CliRunner
from pathlib import Path # Missing import
from typing import Optional # For asset_sha256 type hint

from dam.cli import app # main Typer app
from dam.core.world import World
from dam.services import character_service, ecs_service
from dam.models.conceptual import CharacterConceptComponent, EntityCharacterLinkComponent
from dam.models.properties import FilePropertiesComponent # For creating dummy assets

# Fixture to ensure a clean world for each test function in this file
@pytest.fixture(autouse=True)
async def current_test_world(test_world_alpha: World):
    # The test_world_alpha fixture already handles setup and teardown of the world and its DB.
    # We just need to make it available or ensure it's the one used by CLI context.
    # The CLI uses global_state.world_name, which is set based on --world or default.
    # For these tests, we'll ensure the CLI commands run against 'test_world_alpha'.

    # The settings_override fixture (used by test_world_alpha) sets 'test_world_alpha' as default.
    # So, CLI commands without --world should use it.
    yield test_world_alpha


import asyncio # Add import for asyncio

# @pytest.mark.asyncio # Removed
def test_cli_character_create(current_test_world: World, click_runner: CliRunner): # Removed async
    world_name = current_test_world.name
    char_name = "CLI Test Char 1"
    char_desc = "A character created via CLI for testing."

    result = click_runner.invoke(app, ["--world", world_name, "character", "create", "--name", char_name, "--desc", char_desc], catch_exceptions=False)
    print(f"CLI character create output: {result.stdout}")
    assert result.exit_code == 0
    assert f"Character concept '{char_name}'" in result.stdout
    assert "created successfully" in result.stdout

    char_entity_id = None # To store char_entity.id for the second part

    async def db_checks_after_create():
        nonlocal char_entity_id
        async with current_test_world.db_session_maker() as session:
            char_entity = await character_service.get_character_concept_by_name(session, char_name)
            assert char_entity is not None
            char_entity_id = char_entity.id # Store for later use
            char_comp = await ecs_service.get_component(session, char_entity.id, CharacterConceptComponent)
            assert char_comp is not None
            assert char_comp.concept_name == char_name
            assert char_comp.concept_description == char_desc

    asyncio.run(db_checks_after_create())

    # Test creating again (should indicate existence or handle gracefully)
    # Ensure char_entity_id is available from the async check
    assert char_entity_id is not None, "Character entity ID was not captured from DB checks"
    result_again = click_runner.invoke(app, ["--world", world_name, "character", "create", "--name", char_name])
    assert result_again.exit_code == 0 # Service function handles this gracefully by returning existing
    assert f"Character concept '{char_name}' might already exist" in result_again.stdout or \
           f"Character concept '{char_name}' (Entity ID: {char_entity_id}) created successfully" in result_again.stdout


    # Test validation (e.g., empty name)
    result_empty_name = click_runner.invoke(app, ["--world", world_name, "character", "create", "--name", ""])
    assert result_empty_name.exit_code != 0 # Should fail due to ValueError in service
    assert "Error: Character name cannot be empty." in result_empty_name.stdout


# @pytest.mark.asyncio # Removed
def test_cli_character_apply_list_find(current_test_world: World, sample_text_file: str, click_runner: CliRunner): # Removed async
    world_name = current_test_world.name

    # 1. Create a character
    char_name = "Linkable CLI Char"
    click_runner.invoke(app, ["--world", world_name, "character", "create", "--name", char_name])

    char_id_str_from_db: Optional[str] = None # Variable to store char_id_str

    async def get_char_id():
        nonlocal char_id_str_from_db
        async with current_test_world.db_session_maker() as session:
            char_entity = await character_service.get_character_concept_by_name(session, char_name)
            assert char_entity is not None
            char_id_str_from_db = str(char_entity.id)

    asyncio.run(get_char_id())
    assert char_id_str_from_db is not None # Ensure it was set

    # 2. Add a dummy asset
    asset_id_str_from_db: Optional[str] = None # Variable to store asset_id_str
    async def create_asset_entity():
        nonlocal asset_id_str_from_db
        async with current_test_world.db_session_maker() as session:
            asset_entity = await ecs_service.create_entity(session)
            await ecs_service.add_component_to_entity(
                session, asset_entity.id, FilePropertiesComponent(original_filename="asset_for_char_link.txt")
            )
            await session.commit() # Commit to ensure asset_entity.id is available
            asset_id_str_from_db = str(asset_entity.id)

    asyncio.run(create_asset_entity())
    assert asset_id_str_from_db is not None # Ensure it was set


    # 3. Apply character to asset (using character name and asset ID)
    role = "Main Protagonist"
    result_apply = click_runner.invoke(app, [
        "--world", world_name, "character", "apply",
            "--asset", asset_id_str_from_db,
            "--character", char_name,
        "--role", role
    ])
    print(f"CLI character apply output: {result_apply.stdout}")
    assert result_apply.exit_code == 0
    assert f"Successfully linked character '{char_name}' to asset '{asset_id_str_from_db}' with role '{role}'" in result_apply.stdout

    # Verify link in DB
    async def verify_link_in_db():
        async with current_test_world.db_session_maker() as session:
            links = await ecs_service.get_components(session, int(asset_id_str_from_db), EntityCharacterLinkComponent) # type: ignore
            assert len(links) == 1
            assert links[0].character_concept_entity_id == int(char_id_str_from_db) # type: ignore
            assert links[0].role_in_asset == role

    asyncio.run(verify_link_in_db())

    # 4. List characters for the asset
    result_list = click_runner.invoke(app, ["--world", world_name, "character", "list-for-asset", "--asset", asset_id_str_from_db])
    print(f"CLI character list-for-asset output: {result_list.stdout}")
    assert result_list.exit_code == 0
    assert f"Characters linked to asset '{asset_id_str_from_db}'" in result_list.stdout
    assert f"- {char_name} (Concept ID: {char_id_str_from_db})" in result_list.stdout
    assert f"(Role: {role})" in result_list.stdout

    # 5. Find assets for the character (using character ID)
    result_find = click_runner.invoke(app, ["--world", world_name, "character", "find-assets", "--character", char_id_str_from_db])
    print(f"CLI character find-assets output: {result_find.stdout}")
    assert result_find.exit_code == 0
    assert f"Assets linked to character '{char_name}'" in result_find.stdout
    assert f"Asset ID: {asset_id_str_from_db}" in result_find.stdout
    assert "asset_for_char_link.txt" in result_find.stdout

    # 6. Find assets for character with role filter
    result_find_role = click_runner.invoke(app, [
        "--world", world_name, "character", "find-assets",
        "--character", char_name, "--role", role
    ])
    assert result_find_role.exit_code == 0
    assert f"Asset ID: {asset_id_str_from_db}" in result_find_role.stdout

    # Test finding with non-existent role
    result_find_wrong_role = click_runner.invoke(app, [
        "--world", world_name, "character", "find-assets",
        "--character", char_name, "--role", "NonExistentRole"
    ])
    assert result_find_wrong_role.exit_code == 0
    assert "No assets found for character" in result_find_wrong_role.stdout


# @pytest.mark.asyncio # Removed
def test_cli_character_apply_with_identifiers(current_test_world: World, sample_image_a: Path, click_runner: CliRunner): # Removed async
    # This test uses asset SHA256 hash and character name for identification
    world_name = current_test_world.name

    # 1. Add an asset to get its SHA256 hash
    # Using the CLI to add an asset to ensure it's properly ingested with hashes
    add_result = click_runner.invoke(app, ["--world", world_name, "add-asset", str(sample_image_a)])
    assert add_result.exit_code == 0

    asset_sha256_from_db: Optional[str] = None # Renamed to avoid conflict if asset_sha256 was a parameter

    async def get_asset_sha256():
        nonlocal asset_sha256_from_db
        async with current_test_world.db_session_maker() as session:
            # Find the asset by filename (assuming it's unique for this test)
            entities = await ecs_service.find_entities_by_component_attribute_value(
                session, FilePropertiesComponent, "original_filename", sample_image_a.name
            )
            assert len(entities) == 1
            asset_entity = entities[0]
            asset_id_for_test = asset_entity.id

            from dam.models.hashes import ContentHashSHA256Component
            sha_comp = await ecs_service.get_component(session, asset_id_for_test, ContentHashSHA256Component)
            assert sha_comp is not None
            asset_sha256_from_db = sha_comp.hash_value.hex()

    asyncio.run(get_asset_sha256())
    assert asset_sha256_from_db is not None

    # 2. Create a character
    char_name_for_hash_test = "CharForHashAssetTest"
    click_runner.invoke(app, ["--world", world_name, "character", "create", "--name", char_name_for_hash_test])

    # 3. Apply character using asset hash and character name
    result_apply_hash = click_runner.invoke(app, [
        "--world", world_name, "character", "apply",
        "--asset", asset_sha256_from_db,
        "--character", char_name_for_hash_test
    ])
    print(f"CLI apply with hash output: {result_apply_hash.stdout}")
    assert result_apply_hash.exit_code == 0
    assert f"Successfully linked character '{char_name_for_hash_test}' to asset '{asset_sha256_from_db}'" in result_apply_hash.stdout

    # 4. List characters for asset using asset hash
    result_list_hash = click_runner.invoke(app, ["--world", world_name, "character", "list-for-asset", "--asset", asset_sha256_from_db])
    print(f"CLI list with hash output: {result_list_hash.stdout}")
    assert result_list_hash.exit_code == 0
    assert f"Characters linked to asset '{asset_sha256_from_db}'" in result_list_hash.stdout
    assert char_name_for_hash_test in result_list_hash.stdout
