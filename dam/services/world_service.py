import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Type  # Removed List

from sqlalchemy.inspection import inspect as sqlalchemy_inspect
from sqlalchemy.orm import Session, joinedload

# Use REGISTERED_COMPONENT_TYPES from base_component, which is populated by __init_subclass__
from dam.models.base_component import REGISTERED_COMPONENT_TYPES, BaseComponent
from dam.models.entity import Entity
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


def export_ecs_world_to_json(session: Session, filepath: Path, world_name_for_log: Optional[str] = "current") -> None:
    """
    Exports all entities and their components from the given session to a JSON file.
    Each entity will have its ID and a list of its components.
    Args:
        session: The SQLAlchemy session for the world to be exported.
        filepath: The path to the JSON file where the world data will be saved.
        world_name_for_log: Name of the world, for logging purposes.
    """
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
            # Components are typically accessed via relationship from Entity,
            # e.g., entity.component_file_properties
            # The relationship name is usually the table name.
            relationship_name = component_class.__tablename__  # type: ignore

            components_on_entity = getattr(entity, relationship_name, [])

            if not isinstance(components_on_entity, list):  # If relationship is one-to-one
                components_on_entity = [components_on_entity] if components_on_entity else []

            if components_on_entity:
                component_list_data = []
                for comp_instance in components_on_entity:
                    if comp_instance:  # Ensure component instance is not None
                        comp_data = {
                            c.key: getattr(comp_instance, c.key)
                            for c in sqlalchemy_inspect(comp_instance).mapper.column_attrs
                            if c.key not in ["id", "entity_id"]  # Exclude primary/foreign keys often
                        }
                        # Add discriminator for component type
                        comp_data["__component_type__"] = component_class.__name__
                        component_list_data.append(comp_data)

                if component_list_data:
                    # Store components under their class name for clarity in JSON
                    entity_data["components"][component_class.__name__] = component_list_data

        world_data["entities"].append(entity_data)

    try:
        with open(filepath, "w") as f:
            json.dump(world_data, f, indent=2, default=str)  # default=str for Path, datetime etc.
        logger.info(f"ECS world successfully exported to {filepath}")
    except IOError as e:
        logger.error(f"Error writing ECS world to {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON export: {e}")
        raise


def import_ecs_world_from_json(
    session: Session, filepath: Path, merge: bool = False, world_name_for_log: Optional[str] = "current"
) -> None:
    """
    Imports entities and components from a JSON file into the given session.
    If merge is False (default), it expects a clean database or will raise an error if entities conflict.
    If merge is True, it will try to update existing entities or add new ones. (Merge logic is complex)

    NOTE ON ENTITY IDs:
    When importing entities, if an entity ID from the JSON file does not exist in the target database,
    a new entity is created. This new entity will receive a new, auto-generated ID from the database.
    The original ID from the JSON file is logged but NOT preserved as the primary key for the new entity,
    to maintain compatibility with auto-incrementing primary key sequences.
    If preserving original IDs or mapping them is critical (e.g., for systems requiring stable IDs across
    different DAM instances or for restoring backups to specific ID states), a more sophisticated ID mapping
    and import strategy would be required (see potential future enhancements outlined in comments below).
    """
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
        logger.info(f"ECS world successfully imported from {filepath}")
    except Exception as e:
        session.rollback()
        logger.error(f"Error committing imported ECS world from {filepath}: {e}", exc_info=True)
        raise


# --- Placeholder functions for more advanced operations ---

# The old JSON-based merge_ecs_worlds is removed as it's unused and import_ecs_world_from_json covers its functionality.


def merge_ecs_worlds_db_to_db(
    source_session: Session,
    target_session: Session,
    source_world_name_for_log: Optional[str] = "source",
    target_world_name_for_log: Optional[str] = "target",
    strategy: str = "add_new",  # Currently only 'add_new' is implemented
) -> None:
    """
    Merges entities and components from a source database session to a target database session.
    Strategy 'add_new': All source entities are treated as new in the target.
    """
    logger.info(
        f"Starting DB-to-DB merge from world '{source_world_name_for_log}' to '{target_world_name_for_log}' "
        f"with strategy '{strategy}'."
    )

    if strategy != "add_new":
        logger.error(f"Merge strategy '{strategy}' is not yet implemented. Only 'add_new' is supported.")
        raise NotImplementedError(f"Merge strategy '{strategy}' is not yet implemented.")

    all_component_types = get_all_component_classes()
    source_entities = source_session.query(Entity).all()
    source_entity_count = len(source_entities)
    logger.info(f"Found {source_entity_count} entities in source world '{source_world_name_for_log}'.")

    processed_count = 0
    for src_entity in source_entities:
        processed_count += 1
        logger.debug(f"Processing source entity {src_entity.id} ({processed_count}/{source_entity_count}) for merge...")

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
                    if attr.key not in ["id", "entity_id"]  # These will be set by new association
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

    try:
        target_session.commit()
        logger.info(
            f"Successfully merged {source_entity_count} entities from world '{source_world_name_for_log}' "
            f"to '{target_world_name_for_log}' using 'add_new' strategy."
        )
    except Exception as e:
        target_session.rollback()
        logger.error(f"Error committing merged data to target world '{target_world_name_for_log}': {e}", exc_info=True)
        raise


def split_ecs_world(  # Renamed from placeholder
    source_session: Session,
    criteria: Any,  # This will need to be more specific, e.g. a callable or a structured query
    target_session_selected: Session,
    target_session_remaining: Session,
    source_world_name_for_log: Optional[str] = "source",
    target_selected_world_name_for_log: Optional[str] = "target_selected",
    target_remaining_world_name_for_log: Optional[str] = "target_remaining",
    criteria_component_name: Optional[str] = None,
    criteria_component_attr: Optional[str] = None,
    criteria_value: Optional[Any] = None,
    criteria_op: str = "eq",  # Supported: eq, ne, contains, startswith, endswith (for str); gt, lt, ge, le (for numeric/date)
    delete_from_source: bool = False,
) -> Tuple[int, int]:  # Returns (count_selected, count_remaining)
    """
    Splits entities from a source session into two target sessions based on component criteria.
    Copies entities and their components. Does not delete from source by default.

    TRANSACTION MANAGEMENT NOTE:
    This operation involves multiple database sessions (source, target_selected, target_remaining).
    Commits are performed sequentially for each session. Therefore, the entire split operation
    is NOT ATOMIC across all involved databases.
    - Operations on `target_session_selected` are committed together.
    - Operations on `target_session_remaining` are committed together.
    - If `delete_from_source` is True, deletions on `source_session` are committed together.
    However, a failure in a later commit (e.g., committing deletions from source) will not
    roll back earlier successful commits (e.g., additions to target worlds).
    This can lead to an intermediate state (e.g., entities copied but not deleted from source).
    Users should be aware of this, especially when using `delete_from_source=True`.
    For critical operations, consider backups or manual verification steps.
    """
    logger.info(
        f"Starting DB-to-DB split from world '{source_world_name_for_log}' "
        f"to selected='{target_selected_world_name_for_log}', remaining='{target_remaining_world_name_for_log}'."
    )
    if criteria_component_name and criteria_component_attr and criteria_value is not None:
        logger.info(
            f"Split criteria: Component='{criteria_component_name}', Attribute='{criteria_component_attr}', "
            f"Operator='{criteria_op}', Value='{criteria_value}'"
        )
    else:
        logger.warning("No split criteria provided. All entities will go to 'remaining' target.")
        # Or raise error, or handle as "select none"

    all_component_types = get_all_component_classes()
    source_entities = source_session.query(Entity).all()
    source_entity_count = len(source_entities)
    logger.info(f"Found {source_entity_count} entities in source world '{source_world_name_for_log}'.")

    # Find the criteria component class
    CriteriaComponentClass: Optional[Type[BaseComponent]] = None
    if criteria_component_name:
        for comp_class in all_component_types:
            if comp_class.__name__ == criteria_component_name:
                CriteriaComponentClass = comp_class
                break
        if not CriteriaComponentClass:
            logger.error(f"Criteria component class '{criteria_component_name}' not found.")
            raise ValueError(f"Criteria component class '{criteria_component_name}' not found.")
        if criteria_component_attr and not hasattr(CriteriaComponentClass, criteria_component_attr):
            logger.error(
                f"Criteria attribute '{criteria_component_attr}' not found in component '{criteria_component_name}'."
            )
            raise ValueError(
                f"Criteria attribute '{criteria_component_attr}' not found in component '{criteria_component_name}'."
            )

    count_selected = 0
    count_remaining = 0

    for src_entity in source_entities:
        matches_criteria = False
        if CriteriaComponentClass and criteria_component_attr and criteria_value is not None:
            # Get the specific component instances for the source entity
            # Assuming single component instance for criteria for simplicity now.
            # If multiple components of this type can exist, logic needs to decide (any match? all match?)
            src_criteria_comp_instance = ecs_service.get_component(
                source_session, src_entity.id, CriteriaComponentClass
            )
            if src_criteria_comp_instance:
                actual_value = getattr(src_criteria_comp_instance, criteria_component_attr, None)
                if actual_value is not None:
                    # Perform comparison based on criteria_op
                    if criteria_op == "eq":
                        matches_criteria = actual_value == criteria_value
                    elif criteria_op == "ne":
                        matches_criteria = actual_value != criteria_value
                    elif criteria_op == "contains" and isinstance(actual_value, str):
                        matches_criteria = criteria_value in actual_value
                    elif criteria_op == "startswith" and isinstance(actual_value, str):
                        matches_criteria = actual_value.startswith(criteria_value)
                    elif criteria_op == "endswith" and isinstance(actual_value, str):
                        matches_criteria = actual_value.endswith(criteria_value)
                    # Basic numeric/date ops (assuming criteria_value is already correct type or castable)
                    elif criteria_op == "gt":
                        matches_criteria = actual_value > criteria_value
                    elif criteria_op == "lt":
                        matches_criteria = actual_value < criteria_value
                    elif criteria_op == "ge":
                        matches_criteria = actual_value >= criteria_value
                    elif criteria_op == "le":
                        matches_criteria = actual_value <= criteria_value
                    else:
                        logger.warning(f"Unsupported criteria operator '{criteria_op}'. Defaulting to no match.")

        target_session_for_copy = target_session_selected if matches_criteria else target_session_remaining
        target_world_log_name = (
            target_selected_world_name_for_log if matches_criteria else target_remaining_world_name_for_log
        )

        if matches_criteria:
            count_selected += 1
        else:
            count_remaining += 1

        logger.debug(
            f"Entity {src_entity.id}: Criteria match={matches_criteria}. Copying to '{target_world_log_name}'."
        )

        # Create new entity in the chosen target session
        tgt_entity = ecs_service.create_entity(target_session_for_copy)  # Flushes session

        # Copy components (same logic as merge_ecs_worlds_db_to_db)
        for ComponentClassToCopy in all_component_types:
            src_components_to_copy = ecs_service.get_components(source_session, src_entity.id, ComponentClassToCopy)
            for src_comp_instance in src_components_to_copy:
                comp_data = {
                    attr.key: getattr(src_comp_instance, attr.key)
                    for attr in sqlalchemy_inspect(src_comp_instance).mapper.column_attrs
                    if attr.key not in ["id", "entity_id"]
                }
                comp_data_for_new = comp_data.copy()
                comp_data_for_new["entity_id"] = tgt_entity.id
                comp_data_for_new["entity"] = tgt_entity

                try:
                    new_comp_instance = ComponentClassToCopy(**comp_data_for_new)
                    ecs_service.add_component_to_entity(
                        target_session_for_copy, tgt_entity.id, new_comp_instance, flush=False
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to copy component {ComponentClassToCopy.__name__} for src_entity {src_entity.id} "
                        f"to tgt_entity {tgt_entity.id} in world '{target_world_log_name}': {e}",
                        exc_info=True,
                    )

        if delete_from_source:
            logger.debug(f"Deleting entity {src_entity.id} from source world '{source_world_name_for_log}'.")
            # Pass flush=False as we'll commit source_session at the end.
            ecs_service.delete_entity(source_session, src_entity.id, flush=False)

    try:
        target_session_selected.commit()
        logger.info(
            f"Committed {count_selected} entities to target_selected world '{target_selected_world_name_for_log}'."
        )
    except Exception as e:
        target_session_selected.rollback()
        logger.error(
            f"Error committing to target_selected world '{target_selected_world_name_for_log}': {e}", exc_info=True
        )
        # Potentially raise or handle so remaining isn't committed either if atomicity is desired across all targets
        raise

    try:
        target_session_remaining.commit()
        logger.info(
            f"Committed {count_remaining} entities to target_remaining world '{target_remaining_world_name_for_log}'."
        )
    except Exception as e:
        target_session_remaining.rollback()
        logger.error(
            f"Error committing to target_remaining world '{target_remaining_world_name_for_log}': {e}", exc_info=True
        )
        raise

    if delete_from_source:
        try:
            source_session.commit()
            logger.info(
                f"Committed deletions of {source_entity_count} entities from source world '{source_world_name_for_log}'."
            )
        except Exception as e:
            source_session.rollback()
            logger.error(
                f"Error committing deletions from source world '{source_world_name_for_log}': {e}", exc_info=True
            )
            # This is problematic as entities might already be copied.
            # Full transactional split across DBs is very hard.
            raise

    logger.info(
        f"Split complete: {count_selected} entities to '{target_selected_world_name_for_log}', "
        f"{count_remaining} entities to '{target_remaining_world_name_for_log}'."
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
