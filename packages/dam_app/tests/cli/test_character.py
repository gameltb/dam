"""Tests for the character-related CLI commands and service functions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from dam.core.database import DatabaseManager
from dam.functions import character_functions as character_service
from dam.functions import ecs_functions as ecs_service
from dam.models.conceptual import CharacterConceptComponent, EntityCharacterLinkComponent
from dam.models.hashes import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.models.filename_component import FilenameComponent
from dam_fs.settings import FsSettingsComponent
from dam_test_utils.types import WorldFactory


@pytest.mark.asyncio
async def test_cli_character_create(world_factory: WorldFactory):
    """Test the creation of character concepts via the service layer."""
    world = await world_factory("test_world", [])
    async with world.get_resource(DatabaseManager).get_db_session() as db_session:
        char_name = "CLI Test Char 1"
        char_desc = "A character created via CLI for testing."

        # Initial creation
        char_entity = await character_service.create_character_concept(
            session=db_session, name=char_name, description=char_desc
        )
        assert char_entity is not None
        char_entity_id = char_entity.id

        # DB checks for initial creation
        retrieved_char_entity = await character_service.get_character_concept_by_name(db_session, char_name)
        assert retrieved_char_entity is not None
        assert retrieved_char_entity.id == char_entity_id
        char_comp = await ecs_service.get_component(db_session, retrieved_char_entity.id, CharacterConceptComponent)
        assert char_comp is not None
        assert char_comp.concept_name == char_name
        assert char_comp.concept_description == char_desc

        # Test creating again (should return existing or handle gracefully by service)
        # The service function create_character_concept logs a warning and returns existing if found.
        existing_char_entity_again = await character_service.create_character_concept(
            session=db_session,
            name=char_name,
            description="Different desc",  # Try different desc
        )
        assert existing_char_entity_again is not None
        assert existing_char_entity_again.id == char_entity_id  # Should be the same entity

        # Verify that the description was NOT updated (service returns existing, doesn't update)
        char_comp_after_dupe_attempt = await ecs_service.get_component(
            db_session, char_entity_id, CharacterConceptComponent
        )
        assert char_comp_after_dupe_attempt is not None
        assert char_comp_after_dupe_attempt.concept_description == char_desc  # Original description

        # Test validation (e.g., empty name) by trying to call the service function
        with pytest.raises(ValueError, match=r"Character name cannot be empty\."):
            await character_service.create_character_concept(session=db_session, name="", description="Test empty name")


@pytest.mark.asyncio
async def test_cli_character_apply_list_find(world_factory: WorldFactory):
    """Test applying, listing, and finding characters on assets via the service layer."""
    world = await world_factory("test_world", [])
    async with world.get_resource(DatabaseManager).get_db_session() as db_session:
        # 1. Create a character by directly calling the service
        char_name = "Linkable Service Char"
        char_entity = await character_service.create_character_concept(db_session, name=char_name)
        assert char_entity is not None
        char_id = char_entity.id

        # 2. Add a dummy asset by directly calling the service
        asset_filename = "asset_for_char_link_service.txt"
        asset_entity = await ecs_service.create_entity(db_session)
        await ecs_service.add_component_to_entity(
            db_session,
            asset_entity.id,
            FilenameComponent(filename=asset_filename, first_seen_at=datetime.now(UTC)),
        )
        await ecs_service.add_component_to_entity(
            db_session, asset_entity.id, ContentLengthComponent(file_size_bytes=100)
        )
        asset_id = asset_entity.id

        # 3. Apply character to asset (using service call)
        role = "Main Protagonist Service"
        link_component = await character_service.apply_character_to_entity(
            session=db_session,
            entity_id_to_link=asset_id,
            character_concept_entity_id=char_id,
            role=role,
        )
        assert link_component is not None
        assert link_component.role_in_asset == role

        # Verify link in DB (already part of the above apply logic implicitly, but can be explicit)
        links = await ecs_service.get_components(db_session, asset_id, EntityCharacterLinkComponent)
        assert len(links) == 1
        assert links[0].character_concept_entity_id == char_id
        assert links[0].role_in_asset == role

        # 4. List characters for the asset (using service call)
        characters_on_asset = await character_service.get_characters_for_entity(db_session, asset_id)
        assert len(characters_on_asset) == 1
        found_char_entity, found_role = characters_on_asset[0]
        assert found_char_entity.id == char_id
        assert found_role == role
        char_comp = await ecs_service.get_component(db_session, found_char_entity.id, CharacterConceptComponent)
        assert char_comp is not None
        assert char_comp.concept_name == char_name

        # 5. Find assets for the character (using service call)
        linked_assets = await character_service.get_entities_for_character(db_session, char_id)
        assert len(linked_assets) == 1
        assert linked_assets[0].id == asset_id
        fnc = await ecs_service.get_component(db_session, linked_assets[0].id, FilenameComponent)
        assert fnc is not None
        assert fnc.filename == asset_filename

        # 6. Find assets for character with role filter (using service call)
        linked_assets_role_filter = await character_service.get_entities_for_character(
            db_session, char_id, role_filter=role
        )
        assert len(linked_assets_role_filter) == 1
        assert linked_assets_role_filter[0].id == asset_id

        # Test finding with non-existent role (using service call)
        linked_assets_wrong_role = await character_service.get_entities_for_character(
            db_session, char_id, role_filter="NonExistentRole"
        )
        assert len(linked_assets_wrong_role) == 0


@pytest.mark.asyncio
async def test_cli_character_apply_with_identifiers(world_factory: WorldFactory, temp_image_file: Path, tmp_path: Path):
    """Test applying characters to assets using different identifiers (name, hash)."""
    # This test uses asset SHA256 hash and character name for identification

    # 1. Add an asset to get its SHA256 hash
    add_command = RegisterLocalFileCommand(file_path=temp_image_file)
    world = await world_factory(
        "test_world",
        [
            FsSettingsComponent(
                plugin_name="dam-fs",
                asset_storage_path=str(tmp_path),
            )
        ],
    )
    await world.dispatch_command(add_command).get_all_results()

    async with world.get_resource(DatabaseManager).get_db_session() as session:
        entities = await ecs_service.find_entities_by_component_attribute_value(
            session, FilenameComponent, "filename", temp_image_file.name
        )
        assert len(entities) == 1
        asset_entity = entities[0]
        asset_id_for_test = asset_entity.id

        sha_comp = await ecs_service.get_component(session, asset_id_for_test, ContentHashSHA256Component)
        assert sha_comp is not None

        char_name_for_hash_test = "CharForHashAssetTestSvc"
        char_entity = await character_service.create_character_concept(session, name=char_name_for_hash_test)
        assert char_entity is not None
        char_id_for_hash_test = char_entity.id

        # 3. Apply character using asset ID and character ID (service call)
        link_component = await character_service.apply_character_to_entity(
            session=session,
            entity_id_to_link=asset_id_for_test,
            character_concept_entity_id=char_id_for_hash_test,
            role=None,  # No role specified in this test part
        )
        assert link_component is not None

        # Verify in DB
        links = await ecs_service.get_components(session, asset_id_for_test, EntityCharacterLinkComponent)
        assert len(links) == 1
        assert links[0].character_concept_entity_id == char_id_for_hash_test

        # 4. List characters for asset using asset ID (service call)
        characters_on_asset = await character_service.get_characters_for_entity(session, asset_id_for_test)
        assert len(characters_on_asset) == 1
        found_char_entity, _ = characters_on_asset[0]
        assert found_char_entity.id == char_id_for_hash_test

        char_comp = await ecs_service.get_component(session, found_char_entity.id, CharacterConceptComponent)
        assert char_comp is not None
        assert char_comp.concept_name == char_name_for_hash_test
