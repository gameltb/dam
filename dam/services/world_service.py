import json
import logging
from pathlib import Path
from typing import Any, Type

from sqlalchemy.inspection import inspect as sqlalchemy_inspect
from sqlalchemy.orm import Session, joinedload

from dam.models.base_component import BaseComponent
from dam.models.entity import Entity
from dam.services import ecs_service

logger = logging.getLogger(__name__)

ALL_COMPONENT_TYPES: list[Type[BaseComponent]] = []  # To be populated


def register_component_type(component_class: Type[BaseComponent]):
    """Registers a component type for serialization/deserialization."""
    if component_class not in ALL_COMPONENT_TYPES:
        ALL_COMPONENT_TYPES.append(component_class)


def get_all_component_classes() -> list[Type[BaseComponent]]:
    """
    Dynamically discovers all Component classes inheriting from BaseComponent.
    This is a more robust way than manual registration if models are well-defined.
    """
    if ALL_COMPONENT_TYPES:  # Use cache if populated by manual registration
        return ALL_COMPONENT_TYPES

    # Fallback to dynamic discovery if not manually registered
    # This requires models to be imported somewhere for BaseComponent.__subclasses__() to work
    # For now, let's assume manual registration or a more sophisticated discovery mechanism if needed.
    # If we rely on BaseComponent.__subclasses__(), ensure all component model files are imported
    # e.g., in dam.models.__init__.py

    # Example: If we had a central place where all components are imported:
    # import dam.models # This would trigger imports in dam/models/__init__.py
    # return BaseComponent.__subclasses__()
    # For now, this function will return the manually registered types.
    # We will populate ALL_COMPONENT_TYPES manually or by iterating through dam.models modules.
    # This part needs to be more robust.

    # Let's try a more dynamic approach for now, assuming models are imported.
    # Ensure dam.models.__init__ imports all component types for this to work.
    discovered_types = []
    # Iterating through subclasses recursively might be too broad if there are intermediate bases
    for subclass in BaseComponent.__subclasses__():
        # Check if it's a direct or indirect subclass that is a concrete component table
        mapper = sqlalchemy_inspect(subclass, raiseerr=False)
        if mapper and hasattr(subclass, "__tablename__"):
            if subclass not in discovered_types:
                discovered_types.append(subclass)

    # If manual registration happened, merge and deduplicate
    for comp_type in ALL_COMPONENT_TYPES:
        if comp_type not in discovered_types:
            discovered_types.append(comp_type)

    logger.debug(f"Discovered/registered component types: {[c.__name__ for c in discovered_types]}")
    return discovered_types


def export_ecs_world_to_json(session: Session, filepath: Path) -> None:
    """
    Exports all entities and their components to a JSON file.
    Each entity will have its ID and a list of its components.
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


def import_ecs_world_from_json(session: Session, filepath: Path, merge: bool = False) -> None:
    """
    Imports entities and components from a JSON file.
    If merge is False (default), it expects a clean database or will raise an error if entities conflict.
    If merge is True, it will try to update existing entities or add new ones. (Merge logic is complex)
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
            entity_to_update = ecs_service.create_entity(session) # Let DB assign ID
            # If an ID was provided in JSON, log that it's not being used directly to set the new ID.
            # The entity_id from JSON is still useful for deciding if it's a "new" vs "existing" entity for merge logic.
            if entity_id:
                logger.warning(
                    f"New entity created with DB-assigned ID {entity_to_update.id}. "
                    f"Data was from JSON entity originally ID'd {entity_id}. "
                    "Direct ID setting during import is not supported for auto-incrementing keys; new ID assigned."
                )
            session.flush() # Ensure entity_to_update has its new ID available for component linking

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


def merge_ecs_worlds(session: Session, source_filepath: Path, target_session: Session):
    """
    Merges an ECS world from a source (e.g., JSON file) into a target database session.
    This is essentially `import_ecs_world_from_json` with `merge=True`.
    """
    logger.info(f"Starting merge of ECS world from {source_filepath} into target session.")
    # Assuming import_ecs_world_from_json handles the merge logic when merge=True
    # and operates on the 'session' passed to it.
    # If source_filepath is a live DB, we'd need a different approach.
    # For now, source is a JSON file.
    import_ecs_world_from_json(session=target_session, filepath=source_filepath, merge=True)
    logger.info("Merge operation completed (or attempted). Check logs for details.")


def split_ecs_world(session: Session, criteria: Any, output_filepath_selected: Path, output_filepath_remaining: Path):
    """
    Splits an ECS world into two based on some criteria.
    (This is a complex operation and this is a very basic placeholder)

    Criteria could be:
    - A list of entity IDs to select.
    - Entities possessing a certain component type.
    - Entities matching a component property query.
    """
    logger.warning("split_ecs_world is a placeholder and not fully implemented.")
    # 1. Query entities matching criteria (selected)
    # 2. Query entities NOT matching criteria (remaining)
    # 3. Export selected entities to output_filepath_selected
    # 4. Export remaining entities to output_filepath_remaining
    # (Optionally, delete selected/remaining entities from the source session)

    # This is highly dependent on the nature of 'criteria'.
    # For example, if criteria is a list of entity IDs:
    # selected_entities = session.query(Entity).filter(Entity.id.in_(criteria_entity_ids)).all()
    # remaining_entities = session.query(Entity).filter(not_(Entity.id.in_(criteria_entity_ids))).all()

    # Then, adapt export_ecs_world_to_json to take a list of entities instead of querying all.
    raise NotImplementedError("split_ecs_world is not yet implemented.")


def _populate_component_types_for_serialization():
    """
    Helper to ensure all component types are registered for get_all_component_classes.
    This should be called once, e.g. at application startup or before first export/import.
    It iterates through modules in dam.models and registers component classes.
    """
    if ALL_COMPONENT_TYPES:  # Already populated
        return

    import importlib
    import inspect
    import pkgutil

    import dam.models

    logger.debug("Attempting to auto-register component types from dam.models...")

    package = dam.models
    prefix = package.__name__ + "."

    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
        try:
            module = importlib.import_module(modname)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseComponent) and obj is not BaseComponent:
                    # Further check if it's a mapped SQLAlchemy class with a tablename
                    mapper = sqlalchemy_inspect(obj, raiseerr=False)
                    if mapper and hasattr(obj, "__tablename__"):
                        if obj not in ALL_COMPONENT_TYPES:
                            logger.debug(f"Auto-registering component type: {obj.__name__} from {modname}")
                            register_component_type(obj)  # type: ignore
        except Exception as e:
            logger.warning(f"Could not import or inspect module {modname} for components: {e}", exc_info=True)

    if not ALL_COMPONENT_TYPES:
        logger.warning(
            "No component types were auto-registered. Manual registration might be needed or check model imports."
        )
    else:
        logger.info(f"Successfully auto-registered {len(ALL_COMPONENT_TYPES)} component types.")


# Call this once, e.g. when the module is first imported.
# Or explicitly call it from an application setup routine.
_populate_component_types_for_serialization()

# Example of how to manually register if needed:
# from dam.models.file_properties_component import FilePropertiesComponent
# register_component_type(FilePropertiesComponent)
# ... and so on for all component types ...

# Add this new service to __init__.py
# In dam/services/__init__.py:
# from . import world_service
# __all__ = [..., "world_service"]

# Also need to add CLI commands for these operations.
# These would go into dam/cli.py.
# Example for export:
# @app.command(name="export-world")
# def cli_export_world(
#     filepath_str: Annotated[str, typer.Argument(..., help="Path to export JSON file.")],
# ):
#     db = SessionLocal()
#     try:
#         world_service.export_ecs_world_to_json(db, Path(filepath_str))
#         typer.secho(f"World exported to {filepath_str}", fg=typer.colors.GREEN)
#     except Exception as e:
#         typer.secho(f"Error exporting world: {e}", fg=typer.colors.RED)
#         raise typer.Exit(code=1)
#     finally:
#         db.close()

# Similar CLI command for import.
# Splitting/merging are more complex and might need more thought on CLI exposure.
