import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Type  # Removed List

from sqlalchemy.inspection import inspect as sqlalchemy_inspect
from sqlalchemy.orm import Session, joinedload

# Use REGISTERED_COMPONENT_TYPES from base_component, which is populated by __init_subclass__
from dam.models.base_component import REGISTERED_COMPONENT_TYPES, BaseComponent
from dam.models.entity import Entity
from dam.models.file_location_component import FileLocationComponent  # Added import
from dam.services import ecs_service

# No longer need db_manager here if session is always passed in.
# from dam.core.database import db_manager

logger = logging.getLogger(__name__)

# The global ALL_COMPONENT_TYPES and manual registration functions are removed.
# We will rely on REGISTERED_COMPONENT_TYPES from dam.models.base_component.
# _populate_component_types_for_serialization is also removed for the same reason.


def get_all_component_classes() -> list[Type[BaseComponent]]:
    """
    Returns all discovered Component classes that are registered.
    Relies on REGISTERED_COMPONENT_TYPES being populated by metaclass or __init_subclass__ logic.
    """
    if not REGISTERED_COMPONENT_TYPES:
        logger.warning(
            "REGISTERED_COMPONENT_TYPES is empty. "
            "Ensure component models are imported and BaseComponent's registration mechanism is active."
        )
    return REGISTERED_COMPONENT_TYPES


# Forward declaration for World type hint
if False:
    from dam.core.world import World


def export_ecs_world_to_json(export_world: "World", filepath: Path) -> None:
    """
    Exports all entities and their components from the given World to a JSON file.
    Each entity will have its ID and a list of its components.
    Args:
        export_world: The World instance to be exported.
        filepath: The path to the JSON file where the world data will be saved.
    """
    logger.info(f"Starting export of world '{export_world.name}' to {filepath}...")
    session = export_world.get_db_session()
    world_data: dict[str, Any] = {"entities": []}
    component_classes = get_all_component_classes()
    if not component_classes:
        logger.warning("No component types registered or discovered. Export will be empty of components.")

    entities = (
        session.query(Entity)
        .options(
            *[
                joinedload(getattr(Entity, comp_model.__tablename__))
                for comp_model in component_classes
                if hasattr(Entity, comp_model.__tablename__)
            ]
        )
        .all()
    )

    logger.info(f"Found {len(entities)} entities to export.")

    for entity in entities:
        entity_data: dict[str, Any] = {"id": entity.id, "components": {}}

        for component_class in component_classes:
            # Directly query components for the current entity and component_class
            # This is more robust than relying on pre-configured relationships on the Entity model via tablename.
            components_on_entity = session.query(component_class).filter(component_class.entity_id == entity.id).all()

            if components_on_entity:
                component_list_data = []
                for comp_instance in components_on_entity:
                    # comp_instance should not be None if returned from query
                    comp_data = {
                        c.key: getattr(comp_instance, c.key)
                        for c in sqlalchemy_inspect(comp_instance).mapper.column_attrs
                        if c.key not in ["id", "entity_id"]  # Exclude primary/foreign keys
                    }
                    comp_data["__component_type__"] = component_class.__name__
                    component_list_data.append(comp_data)

                if component_list_data:
                    entity_data["components"][component_class.__name__] = component_list_data

        world_data["entities"].append(entity_data)

    try:
        with open(filepath, "w") as f:
            json.dump(world_data, f, indent=2, default=str)
        logger.info(f"World '{export_world.name}' successfully exported to {filepath}")
    except IOError as e:
        logger.error(f"Error writing world '{export_world.name}' to {filepath}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON export for world '{export_world.name}': {e}", exc_info=True)
        raise
    finally:
        session.close()


def import_ecs_world_from_json(
    target_world: "World", filepath: Path, merge: bool = False
) -> None:
    """
    Imports entities and components from a JSON file into the given target World.
    If merge is False (default), it expects a clean database for the target World,
    or will raise an error if entities conflict (behavior might depend on DB constraints).
    If merge is True, it will try to update existing entities or add new ones.

    NOTE ON ENTITY IDs: See notes in original function, behavior regarding new IDs remains.
    """
    logger.info(f"Starting import to world '{target_world.name}' from {filepath} (merge={merge})...")
    session = target_world.get_db_session()
    try:
        with open(filepath, "r") as f:
            world_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Export file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred reading export file: {e}")
        raise

    if "entities" not in world_data:
        logger.error("Invalid export format: 'entities' key missing.")
        raise ValueError("Invalid export format: 'entities' key missing.")

    component_name_to_class_map = {cls.__name__: cls for cls in get_all_component_classes()}
    if not component_name_to_class_map:
        logger.warning("No component types registered/discovered. Import may not restore components correctly.")

    # --- Potential Future Enhancement: ID Mapping for Imports ---
    # To implement robust ID preservation or mapping, especially when dealing with
    # auto-incrementing primary keys or when needing to resolve ID conflicts more gracefully:
    #
    # 1. First Pass (Dry Run or Pre-Scan - Optional but Recommended):
    #    - Iterate through all entity IDs in the JSON.
    #    - Check their existence in the target database.
    #    - Build a proposed `id_map: Dict[int_json_id, int_db_id]`.
    #    - For new entities, the `int_db_id` would be a placeholder indicating it needs creation.
    #    - For existing entities (if not merging or if merge strategy allows updates):
    #        - `id_map[json_id] = existing_db_id`.
    #    - This pass can also identify potential ID conflicts if not using auto-assignment for new IDs.
    #
    # 2. Entity Creation and ID Mapping Pass:
    #    - Iterate through JSON entities again.
    #    - If `json_id` is to be a new entity:
    #        - `new_db_entity = ecs_service.create_entity(session)`
    #        - `id_map[json_id] = new_db_entity.id`
    #    - If `json_id` maps to an existing entity (and merging/updating):
    #        - Fetch the entity using `id_map[json_id]`.
    #
    # 3. Component Processing Pass:
    #    - Iterate through JSON entities and their components.
    #    - Use `id_map[json_entity_id]` to get the correct `db_entity_id` for associating components.
    #    - If components themselves have relationships to other entities (not currently the case in this model
    #      where components are simple data containers tied to one entity), those entity ID references
    #      would also need to be translated using the `id_map`.
    #
    # 4. Handling `entity_id` in Component Data:
    #    - The `comp_data_cleaned["entity_id"]` would be set using the mapped `db_entity_id`.
    #
    # This approach decouples JSON IDs from database IDs, essential for auto-increment PKs,
    # and provides a foundation for more complex import/merge strategies.
    # It adds complexity, requiring multiple passes or holding more data in memory during import.
    # The current simpler approach (new entities always get new DB-assigned IDs) is safer
    # for general cases but lacks ID stability if that's a cross-system requirement.
    # --- End of Future Enhancement Outline ---

    for entity_data in world_data["entities"]:
        entity_id = entity_data.get("id")
        if entity_id is None:
            logger.warning("Skipping entity data with no ID.")
            continue

        existing_entity = session.get(Entity, entity_id)

        if existing_entity:
            if not merge:
                logger.error(
                    f"Entity ID {entity_id} already exists in the database. "
                    "Import with merge=False requires a clean target or non-conflicting IDs."
                )
                # Potentially raise an error or skip. For now, skipping.
                # raise ValueError(f"Entity ID {entity_id} already exists.")
                logger.warning(f"Skipping existing Entity ID {entity_id} due to no-merge policy.")
                continue
            else:  # Merge logic for existing entity
                logger.info(f"Merging components for existing Entity ID {entity_id}.")
                entity_to_update = existing_entity
        else:  # New entity
            # We need to be careful if IDs are externally managed or DB auto-increments.
            # If DB auto-increments, we cannot force IDs like this without issues
            # unless identity insert is enabled, or IDs are not primary keys.
            # For this example, let's assume IDs can be set if the entity is new.
            # This might require specific DB configurations (e.g., disabling autoincrement temporarily
            # or ensuring Entity.id is not auto-incrementing if we manage IDs this way).
            # A safer approach for auto-increment IDs would be to map old IDs to new IDs.
            # For now, let's assume we can create with specified ID if it doesn't exist.
            # This is a simplification.
            logger.info(f"Creating new Entity with ID {entity_id}.")
            # This line below is problematic with auto-incrementing PKs.
            # entity_to_update = Entity(id=entity_id)
            # A better way for auto-inc PKs is to create entity, then map old_id -> new_id
            # For now, we will use ecs_service.create_entity() which does not take an ID.
            # This means imported entity IDs will not match original unless we handle ID mapping.
            # This is a significant simplification for now.
            # A proper solution would involve:
            # 1. Create entity (gets new auto-inc ID).
            # 2. Store a map: old_entity_id_from_json -> new_entity_id_in_db.
            # 3. When adding components, use the new_entity_id_in_db.
            # For this iteration, we'll try to use the given ID, assuming it might work or be handled by user.

            # Safer: Create entity and let DB assign ID, then print a mapping or warn.
            # For simplicity of this step, let's assume we try to use the ID from JSON.
            # This will likely fail with standard auto-incrementing primary keys if the ID is not already in sequence.
            # A robust import would handle ID mapping.

            # Create a new entity. The DB will assign a new ID.
            # Original IDs from JSON are not preserved directly with auto-incrementing keys.
            # A more complex system would map old IDs to new IDs.
            logger.info(f"Creating new entity for data originally ID'd as {entity_id} in JSON.")
            entity_to_update = ecs_service.create_entity(session)  # Let DB assign ID
            # If an ID was provided in JSON, log that it's not being used directly to set the new ID.
            # The entity_id from JSON is still useful for deciding if it's a "new" vs "existing" entity for merge logic.
            if entity_id:
                logger.warning(
                    f"New entity created with DB-assigned ID {entity_to_update.id}. "
                    f"Data was from JSON entity originally ID'd {entity_id}. "
                    "Direct ID setting during import is not supported for auto-incrementing keys; new ID assigned."
                )
            session.flush()  # Ensure entity_to_update has its new ID available for component linking

        # Process components for the entity
        components_data = entity_data.get("components", {})
        for comp_type_name, comp_list_data in components_data.items():
            ComponentClass = component_name_to_class_map.get(comp_type_name)
            if not ComponentClass:
                logger.warning(f"Unknown component type '{comp_type_name}' in JSON for entity {entity_id}. Skipping.")
                continue

            if merge and existing_entity:
                # Clear existing components of this type for the entity before adding new ones
                # This is a simple merge strategy: replace. More complex strategies could be:
                # - update existing based on some key
                # - add if not present
                # - leave existing untouched
                logger.debug(f"Merge: Deleting existing '{comp_type_name}' components for entity {entity_id}.")
                existing_components = ecs_service.get_components(session, entity_to_update.id, ComponentClass)
                for comp_to_delete in existing_components:
                    session.delete(comp_to_delete)
                session.flush()  # Apply deletions

            for comp_data in comp_list_data:
                # Remove our type discriminator before passing to constructor
                comp_data_cleaned = {k: v for k, v in comp_data.items() if k != "__component_type__"}
                try:
                    # Ensure all necessary fields for ComponentClass constructor are present
                    # This includes entity_id and the entity relationship if BaseComponent requires it
                    comp_data_cleaned["entity_id"] = entity_to_update.id
                    # If BaseComponent's __init__ truly needs `entity` object due to kw_only:
                    comp_data_cleaned["entity"] = entity_to_update

                    # Map old FileLocationComponent field names from JSON to new model fields if necessary
                    if ComponentClass == FileLocationComponent:
                        if "file_identifier" in comp_data_cleaned and "content_identifier" not in comp_data_cleaned:
                            comp_data_cleaned["content_identifier"] = comp_data_cleaned.pop("file_identifier")
                        if "original_filename" in comp_data_cleaned and "contextual_filename" not in comp_data_cleaned:
                            comp_data_cleaned["contextual_filename"] = comp_data_cleaned.pop("original_filename")
                        # physical_path_or_key should ideally be in the JSON. If it was 'filepath', map it.
                        if (
                            "filepath" in comp_data_cleaned and "physical_path_or_key" not in comp_data_cleaned
                        ):  # Assuming old key might be 'filepath'
                            comp_data_cleaned["physical_path_or_key"] = comp_data_cleaned.pop("filepath")
                        # Ensure physical_path_or_key exists, it's mandatory. This might be an issue if old JSONs don't have it.
                        if "physical_path_or_key" not in comp_data_cleaned:
                            # Try to use content_identifier as a fallback if it's a CAS-like scenario,
                            # or log an error if it's a critical missing piece.
                            # This depends on how JSONs were generated. For now, let's assume it exists or content_identifier is a proxy.
                            # If 'local_reference', physical_path_or_key is the direct path.
                            # If 'local_cas', physical_path_or_key is the relative CAS path (e.g. aa/bb/hash).
                            # The JSON should reflect this correctly.
                            # If old JSONs used 'file_identifier' for content and also as part of path for CAS,
                            # or 'filepath' for referenced files, the mapping must be robust.
                            # For now, we assume the new field names are in JSON or mapped above.
                            # If physical_path_or_key is still missing, it's an issue.
                            logger.warning(
                                f"FileLocationComponent data for entity {entity_to_update.id} is missing 'physical_path_or_key'. Component might be invalid."
                            )

                    new_component = ComponentClass(**comp_data_cleaned)
                    # session.add(new_component) # add_component_to_entity handles this
                    ecs_service.add_component_to_entity(session, entity_to_update.id, new_component, flush=False)
                    logger.debug(f"Added component {comp_type_name} to entity {entity_to_update.id}")
                except Exception as e:
                    logger.error(
                        f"Failed to create/add component {comp_type_name} for entity {entity_to_update.id} "
                        f"with data {comp_data_cleaned}: {e}",
                        exc_info=True,
                    )
                    # Depending on policy, might rollback or continue

    try:
        session.commit()
        logger.info(f"World '{target_world.name}' successfully imported from {filepath}")
    except Exception as e:
        session.rollback()
        logger.error(f"Error committing imported data to world '{target_world.name}' from {filepath}: {e}", exc_info=True)
        raise
    finally:
        session.close()


# --- Placeholder functions for more advanced operations ---

# The old JSON-based merge_ecs_worlds is removed as it's unused and import_ecs_world_from_json covers its functionality.


def merge_ecs_worlds_db_to_db(
    source_world: "World",
    target_world: "World",
    strategy: str = "add_new",  # Currently only 'add_new' is implemented
) -> None:
    """
    Merges entities and components from a source World's database to a target World's database.
    Strategy 'add_new': All source entities are treated as new in the target.
    """
    logger.info(
        f"Starting DB-to-DB merge from world '{source_world.name}' to '{target_world.name}' "
        f"with strategy '{strategy}'."
    )

    if strategy != "add_new":
        logger.error(f"Merge strategy '{strategy}' for worlds '{source_world.name}' -> '{target_world.name}' is not yet implemented. Only 'add_new' is supported.")
        raise NotImplementedError(f"Merge strategy '{strategy}' is not yet implemented.")

    source_session = source_world.get_db_session()
    target_session = target_world.get_db_session()

    try:
        all_component_types = get_all_component_classes()
        source_entities = source_session.query(Entity).all()
        source_entity_count = len(source_entities)
        logger.info(f"Found {source_entity_count} entities in source world '{source_world.name}'.")

        processed_count = 0
        for src_entity in source_entities:
            processed_count += 1
            logger.debug(f"Processing source entity {src_entity.id} ({processed_count}/{source_entity_count}) for merge to '{target_world.name}'...")

            # Strategy 'add_new': Create a new entity in the target world
            tgt_entity = ecs_service.create_entity(target_session)  # Flushes session

            # Copy components
            for ComponentClass in all_component_types:
                # Get all components of this type for the source entity
                src_components = ecs_service.get_components(source_session, src_entity.id, ComponentClass)

                for src_comp_instance in src_components:
                    # Create a new component instance for the target entity
                    # Copy attribute values, excluding 'id', 'entity_id', and 'entity' relationship
                    comp_data = {
                        attr.key: getattr(src_comp_instance, attr.key)
                        for attr in sqlalchemy_inspect(src_comp_instance).mapper.column_attrs
                        if attr.key not in ["id", "entity_id", "created_at", "updated_at"]
                    }

                    # Add entity_id and entity for the new target entity
                    # Note: kw_only=True on BaseComponent means these must be provided if not default
                    # The ComponentClass(**comp_data) will implicitly use BaseComponent's __init__
                    # which expects entity_id and entity if they are not init=False or have defaults.
                    # We need to ensure ComponentClass can be instantiated with comp_data,
                    # and then add_component_to_entity will correctly link it.
                    # The current BaseComponent requires entity_id at init.

                    # Let's create component with new entity_id and entity, then add
                    comp_data_for_new = comp_data.copy()
                    comp_data_for_new["entity_id"] = tgt_entity.id
                    comp_data_for_new["entity"] = tgt_entity  # For relationship

                    try:
                        new_comp_instance = ComponentClass(**comp_data_for_new)
                        # Add component to the new target entity. Pass flush=False as we commit at the end.
                        ecs_service.add_component_to_entity(target_session, tgt_entity.id, new_comp_instance, flush=False)
                        logger.debug(
                            f"Copied component {ComponentClass.__name__} from src_entity {src_entity.id} "
                            f"to tgt_entity {tgt_entity.id}."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to copy component {ComponentClass.__name__} for src_entity {src_entity.id} "
                            f"to tgt_entity {tgt_entity.id}: {e}",
                            exc_info=True,
                        )
                        # Decide on error handling: rollback transaction, skip component, or skip entity?
                        # For now, log and continue. Transaction commit at the end will either succeed or fail all.

        target_session.commit()
        logger.info(
            f"Successfully merged {source_entity_count} entities from world '{source_world.name}' "
            f"to '{target_world.name}' using 'add_new' strategy."
        )
    except Exception as e:
        target_session.rollback()
        logger.error(f"Error committing merged data to target world '{target_world.name}': {e}", exc_info=True)
        # Re-raise to indicate failure of the operation
        raise
    finally:
        source_session.close()
        target_session.close()


def split_ecs_world(
    source_world: "World",
    target_world_selected: "World",
    target_world_remaining: "World",
    criteria_component_name: Optional[str] = None,
    criteria_component_attr: Optional[str] = None,
    criteria_value: Optional[Any] = None,
    criteria_op: str = "eq",
    delete_from_source: bool = False,
) -> Tuple[int, int]:
    """
    Splits entities from a source World into two target Worlds based on component criteria.
    Copies entities and their components. Optionally deletes from source.

    TRANSACTION MANAGEMENT NOTE: See original function's note. This behavior persists.
    The function now manages sessions internally by acquiring them from the World objects.
    """
    logger.info(
        f"Starting DB-to-DB split from world '{source_world.name}' "
        f"to selected='{target_world_selected.name}', remaining='{target_world_remaining.name}'."
    )
    if criteria_component_name and criteria_component_attr and criteria_value is not None:
        logger.info(
            f"Split criteria for world '{source_world.name}': Component='{criteria_component_name}', "
            f"Attribute='{criteria_component_attr}', Operator='{criteria_op}', Value='{criteria_value}'"
        )
    else:
        logger.warning("No split criteria provided. All entities will go to 'remaining' target.")
        # Or raise error, or handle as "select none"

        # Or raise error, or handle as "select none"

    source_session = source_world.get_db_session()
    target_session_selected_db = target_world_selected.get_db_session()
    target_session_remaining_db = target_world_remaining.get_db_session()

    try:
        all_component_types = get_all_component_classes()
        source_entities = source_session.query(Entity).all()
        source_entity_count = len(source_entities)
        logger.info(f"Found {source_entity_count} entities in source world '{source_world.name}'.")

        # Find the criteria component class
        CriteriaComponentClass: Optional[Type[BaseComponent]] = None
        if criteria_component_name:
            for comp_class in all_component_types:
                if comp_class.__name__ == criteria_component_name:
                    CriteriaComponentClass = comp_class
                    break
            if not CriteriaComponentClass:
                logger.error(f"Criteria component class '{criteria_component_name}' not found for world '{source_world.name}'.")
                raise ValueError(f"Criteria component class '{criteria_component_name}' not found.")
            if criteria_component_attr and not hasattr(CriteriaComponentClass, criteria_component_attr):
                logger.error(
                    f"Criteria attribute '{criteria_component_attr}' not found in component '{criteria_component_name}' for world '{source_world.name}'."
                )
                raise ValueError(
                    f"Criteria attribute '{criteria_component_attr}' not found in component '{criteria_component_name}'."
                )

        count_selected = 0
        count_remaining = 0

        for src_entity in source_entities:
            matches_criteria = False
            if CriteriaComponentClass and criteria_component_attr and criteria_value is not None:
                src_criteria_comp_instance = ecs_service.get_component(
                    source_session, src_entity.id, CriteriaComponentClass
                )
                if src_criteria_comp_instance:
                    actual_value = getattr(src_criteria_comp_instance, criteria_component_attr, None)
                    if actual_value is not None:
                        if criteria_op == "eq": matches_criteria = actual_value == criteria_value
                        elif criteria_op == "ne": matches_criteria = actual_value != criteria_value
                        elif criteria_op == "contains" and isinstance(actual_value, str): matches_criteria = criteria_value in actual_value
                        elif criteria_op == "startswith" and isinstance(actual_value, str): matches_criteria = actual_value.startswith(criteria_value)
                        elif criteria_op == "endswith" and isinstance(actual_value, str): matches_criteria = actual_value.endswith(criteria_value)
                        elif criteria_op == "gt": matches_criteria = actual_value > criteria_value
                        elif criteria_op == "lt": matches_criteria = actual_value < criteria_value
                        elif criteria_op == "ge": matches_criteria = actual_value >= criteria_value
                        elif criteria_op == "le": matches_criteria = actual_value <= criteria_value
                        else: logger.warning(f"Unsupported criteria operator '{criteria_op}' for world '{source_world.name}'. Defaulting to no match.")

            target_session_for_copy_db = target_session_selected_db if matches_criteria else target_session_remaining_db
            target_world_log_name = target_world_selected.name if matches_criteria else target_world_remaining.name

            if matches_criteria: count_selected += 1
            else: count_remaining += 1

            logger.debug(
                f"Entity {src_entity.id} from world '{source_world.name}': Criteria match={matches_criteria}. Copying to '{target_world_log_name}'."
            )
            tgt_entity = ecs_service.create_entity(target_session_for_copy_db)

            for ComponentClassToCopy in all_component_types:
                src_components_to_copy = ecs_service.get_components(source_session, src_entity.id, ComponentClassToCopy)
                for src_comp_instance in src_components_to_copy:
                    comp_data = {
                        attr.key: getattr(src_comp_instance, attr.key)
                        for attr in sqlalchemy_inspect(src_comp_instance).mapper.column_attrs
                        if attr.key not in ["id", "entity_id", "created_at", "updated_at"]
                    }
                    comp_data_for_new = {**comp_data, "entity_id": tgt_entity.id, "entity": tgt_entity}
                    try:
                        new_comp_instance = ComponentClassToCopy(**comp_data_for_new)
                        ecs_service.add_component_to_entity(
                            target_session_for_copy_db, tgt_entity.id, new_comp_instance, flush=False
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to copy component {ComponentClassToCopy.__name__} for src_entity {src_entity.id} (world '{source_world.name}') "
                            f"to tgt_entity {tgt_entity.id} in world '{target_world_log_name}': {e}",
                            exc_info=True,
                        )
            if delete_from_source:
                logger.debug(f"Deleting entity {src_entity.id} from source world '{source_world.name}'.")
                ecs_service.delete_entity(source_session, src_entity.id, flush=False)

        # Commit target sessions first
        target_session_selected_db.commit()
        logger.info(f"Committed {count_selected} entities to target_selected world '{target_world_selected.name}'.")
        target_session_remaining_db.commit()
        logger.info(f"Committed {count_remaining} entities to target_remaining world '{target_world_remaining.name}'.")

        if delete_from_source:
            source_session.commit()
            logger.info(f"Committed deletions of {source_entity_count} entities from source world '{source_world.name}'.")

    except Exception as e: # Broad catch for issues during processing or commits
        logger.error(f"Error during split operation initiated from source world '{source_world.name}': {e}", exc_info=True)
        # Rollback all sessions involved if an error occurs before all commits are done
        source_session.rollback()
        target_session_selected_db.rollback()
        target_session_remaining_db.rollback()
        raise # Re-raise the exception
    finally:
        # Ensure all sessions are closed
        source_session.close()
        target_session_selected_db.close()
        target_session_remaining_db.close()

    logger.info(
        f"Split from world '{source_world.name}' complete: {count_selected} entities to '{target_world_selected.name}', "
        f"{count_remaining} entities to '{target_world_remaining.name}'."
    )
    return count_selected, count_remaining


# Removed _populate_component_types_for_serialization as REGISTERED_COMPONENT_TYPES should be sufficient.

# CLI command examples removed as they will be handled in cli.py.
# Ensure that when these services are called, the correct session for the intended world is passed.
# For example, a CLI command might look like:
# world_name = ctx.obj.world_name # Get from CLI context
# db_session = db_manager.get_db_session(world_name)
# try:
#     world_service.export_ecs_world_to_json(db_session, export_path, world_name_for_log=world_name)
# finally:
#     db_session.close()
