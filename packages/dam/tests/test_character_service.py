import pytest
from dam.models.conceptual import CharacterConceptComponent, EntityCharacterLinkComponent
from dam.services import character_service, ecs_service
from sqlalchemy import select  # Added import for select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_character_concept(db_session: AsyncSession):
    char_name = "Test Character One"
    char_desc = "A character for testing."

    char_entity = await character_service.create_character_concept(db_session, char_name, char_desc)
    assert char_entity is not None
    assert char_entity.id is not None

    char_comp = await ecs_service.get_component(db_session, char_entity.id, CharacterConceptComponent)
    assert char_comp is not None
    assert char_comp.concept_name == char_name
    assert char_comp.concept_description == char_desc

    # Test creating the same character again (should return existing or handle gracefully)
    char_entity_again = await character_service.create_character_concept(
        db_session, char_name, "Updated desc (should not update here)"
    )
    assert char_entity_again is not None
    assert char_entity_again.id == char_entity.id  # Should return the existing one
    char_comp_again = await ecs_service.get_component(db_session, char_entity_again.id, CharacterConceptComponent)
    assert (
        char_comp_again.concept_description == char_desc
    )  # Description should not have changed by calling create again

    with pytest.raises(ValueError, match="Character name cannot be empty"):
        await character_service.create_character_concept(db_session, "")


@pytest.mark.asyncio
async def test_get_character_concept_by_name_and_id(db_session: AsyncSession):
    char_name = "Finder Character"
    char_desc = "Character to be found."
    char_entity = await character_service.create_character_concept(db_session, char_name, char_desc)
    assert char_entity is not None

    found_by_name = await character_service.get_character_concept_by_name(db_session, char_name)
    assert found_by_name is not None
    assert found_by_name.id == char_entity.id

    found_by_id = await character_service.get_character_concept_by_id(db_session, char_entity.id)
    assert found_by_id is not None
    assert found_by_id.id == char_entity.id

    with pytest.raises(character_service.CharacterConceptNotFoundError):
        await character_service.get_character_concept_by_name(db_session, "NonExistentCharacter")

    assert await character_service.get_character_concept_by_id(db_session, -999) is None


@pytest.mark.asyncio
async def test_find_character_concepts(db_session: AsyncSession):
    await character_service.create_character_concept(db_session, "Alpha Char", "Desc A")
    await character_service.create_character_concept(db_session, "Beta Char", "Desc B")
    await character_service.create_character_concept(db_session, "Gamma Person", "Desc C")

    all_chars = await character_service.find_character_concepts(db_session)
    assert len(all_chars) >= 3  # Greater or equal due to potential chars from other tests

    alpha_search = await character_service.find_character_concepts(db_session, query_name="Alpha")
    assert len(alpha_search) == 1
    alpha_comp = await ecs_service.get_component(db_session, alpha_search[0].id, CharacterConceptComponent)
    assert alpha_comp is not None
    assert alpha_comp.concept_name == "Alpha Char"

    char_search = await character_service.find_character_concepts(db_session, query_name="Char")
    # This depends on exact names; ensure test names are distinct enough.
    # If "Test Character One" and "Finder Character" also exist, this count might be higher.
    # Let's make names more specific for this test or clear DB before.
    # For now, assuming only the 3 above match "Char" if other tests use different naming.
    # This is brittle. A better way is to check if the expected ones are present.
    names_found = []
    for e in char_search:
        comp = await ecs_service.get_component(db_session, e.id, CharacterConceptComponent)
        if comp:
            names_found.append(comp.concept_name)

    assert "Alpha Char" in names_found
    assert "Beta Char" in names_found
    assert "Gamma Person" not in names_found  # "Person" does not contain "Char"


@pytest.mark.asyncio
async def test_update_character_concept(db_session: AsyncSession):
    char_entity = await character_service.create_character_concept(db_session, "Updatable Char", "Initial Desc")
    assert char_entity is not None

    updated_comp = await character_service.update_character_concept(
        db_session, char_entity.id, name="Updated Char Name", description="New Description"
    )
    assert updated_comp is not None
    assert updated_comp.concept_name == "Updated Char Name"
    assert updated_comp.concept_description == "New Description"

    # Test name conflict on update
    await character_service.create_character_concept(db_session, "Existing Name For Update Test", "Desc")
    conflict_update = await character_service.update_character_concept(
        db_session, char_entity.id, name="Existing Name For Update Test"
    )
    assert conflict_update is None  # Should fail due to name conflict

    # Test updating non-existent character
    non_existent_update = await character_service.update_character_concept(db_session, -999, name="No Such Char")
    assert non_existent_update is None


@pytest.mark.asyncio
async def test_delete_character_concept(db_session: AsyncSession):
    char_to_delete = await character_service.create_character_concept(db_session, "Deletable Char", "Will be deleted")
    assert char_to_delete is not None

    # Apply this character to an asset to test link deletion
    asset_entity = await ecs_service.create_entity(db_session)
    await character_service.apply_character_to_entity(db_session, asset_entity.id, char_to_delete.id, role="Victim")

    links = await character_service.get_characters_for_entity(db_session, asset_entity.id)
    assert len(links) == 1

    delete_success = await character_service.delete_character_concept(db_session, char_to_delete.id)
    assert delete_success

    with pytest.raises(character_service.CharacterConceptNotFoundError):
        await character_service.get_character_concept_by_name(db_session, "Deletable Char")

    assert await character_service.get_character_concept_by_id(db_session, char_to_delete.id) is None

    # Check if links were deleted
    links_after_delete = await character_service.get_characters_for_entity(db_session, asset_entity.id)
    assert len(links_after_delete) == 0

    assert not await character_service.delete_character_concept(db_session, -999)  # Delete non-existent


@pytest.mark.asyncio
async def test_apply_and_remove_character_from_entity(db_session: AsyncSession):
    char_entity = await character_service.create_character_concept(db_session, "Linkable Char", "For linking tests")
    asset1 = await ecs_service.create_entity(db_session)
    asset2 = await ecs_service.create_entity(db_session)
    assert char_entity and asset1 and asset2

    # Apply with role
    link1 = await character_service.apply_character_to_entity(db_session, asset1.id, char_entity.id, role="Protagonist")
    assert link1 is not None
    assert link1.character_concept_entity_id == char_entity.id
    assert link1.role_in_asset == "Protagonist"

    # Apply without role
    link2 = await character_service.apply_character_to_entity(db_session, asset1.id, char_entity.id, role=None)
    assert link2 is not None
    assert link2.role_in_asset is None

    # Apply to another asset
    link3 = await character_service.apply_character_to_entity(db_session, asset2.id, char_entity.id, role="Antagonist")
    assert link3 is not None

    # Test applying again (should return existing or handle gracefully)
    link1_again = await character_service.apply_character_to_entity(
        db_session, asset1.id, char_entity.id, role="Protagonist"
    )
    assert link1_again is not None
    assert link1_again.id == link1.id  # Should be the same link component

    # Get characters for asset1
    chars_on_asset1 = await character_service.get_characters_for_entity(db_session, asset1.id)
    assert len(chars_on_asset1) == 2
    # Replace None with empty string for sorting, then map back for assertion if needed, or assert against modified list
    roles_for_sorting = [r if r is not None else "" for e, r in chars_on_asset1]
    roles_on_asset1 = sorted(roles_for_sorting)
    # Expected: None (empty string) sorts before "Protagonist"
    assert roles_on_asset1 == ["", "Protagonist"]

    # Get assets for character
    assets_for_char = await character_service.get_entities_for_character(db_session, char_entity.id)
    assert len(assets_for_char) == 2  # asset1, asset2
    asset_ids_for_char = sorted([e.id for e in assets_for_char])
    assert asset_ids_for_char == sorted([asset1.id, asset2.id])

    assets_with_role_protagonist = await character_service.get_entities_for_character(
        db_session, char_entity.id, role_filter="Protagonist"
    )
    assert len(assets_with_role_protagonist) == 1
    assert assets_with_role_protagonist[0].id == asset1.id

    assets_with_no_role = await character_service.get_entities_for_character(
        db_session, char_entity.id, filter_by_role_presence=False
    )
    assert len(assets_with_no_role) == 1
    assert assets_with_no_role[0].id == asset1.id

    assets_with_any_role = await character_service.get_entities_for_character(
        db_session, char_entity.id, filter_by_role_presence=True
    )
    assert len(assets_with_any_role) == 2  # Protagonist on asset1, Antagonist on asset2
    # Check if asset1 and asset2 are among them (order might vary)
    ids_with_any_role = {e.id for e in assets_with_any_role}
    assert asset1.id in ids_with_any_role
    assert asset2.id in ids_with_any_role

    # Remove character with role
    remove_success1 = await character_service.remove_character_from_entity(
        db_session, asset1.id, char_entity.id, role="Protagonist"
    )
    assert remove_success1
    await db_session.commit()  # Commit the deletion

    # Explicitly try to fetch the supposedly deleted component to confirm deletion
    deleted_link_check_stmt = select(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.entity_id == asset1.id,
        EntityCharacterLinkComponent.character_concept_entity_id == char_entity.id,
        EntityCharacterLinkComponent.role_in_asset == "Protagonist",
    )
    result_deleted_check = await db_session.execute(deleted_link_check_stmt)
    still_exists = result_deleted_check.scalar_one_or_none()
    assert still_exists is None, "The specific link component (Protagonist role) should have been deleted."

    chars_on_asset1_after_remove = await character_service.get_characters_for_entity(db_session, asset1.id)
    assert len(chars_on_asset1_after_remove) == 1
    assert chars_on_asset1_after_remove[0][1] is None  # Only the no-role link should remain

    # Remove character without role
    remove_success2 = await character_service.remove_character_from_entity(
        db_session, asset1.id, char_entity.id, role=None
    )
    assert remove_success2
    await db_session.commit()  # Commit this deletion as well

    chars_on_asset1_final = await character_service.get_characters_for_entity(db_session, asset1.id)
    assert len(chars_on_asset1_final) == 0

    # Test removing non-existent link
    assert not await character_service.remove_character_from_entity(
        db_session, asset2.id, char_entity.id, role="NonExistentRole"
    )

    # Test applying to non-existent asset/character
    assert await character_service.apply_character_to_entity(db_session, -999, char_entity.id) is None
    non_char_entity = await ecs_service.create_entity(db_session)  # An entity without CharacterConceptComponent
    assert await character_service.apply_character_to_entity(db_session, asset1.id, non_char_entity.id) is None
    # (The service currently checks if character_concept_entity_id has CharacterConceptComponent via get_character_concept_by_id)


@pytest.mark.asyncio
async def test_get_entities_for_nonexistent_character(db_session: AsyncSession):
    with pytest.raises(character_service.CharacterConceptNotFoundError):
        await character_service.get_entities_for_character(db_session, -999)
