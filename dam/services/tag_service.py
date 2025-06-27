import logging
from typing import List, Optional, Tuple, Type

from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from dam.models.core.entity import Entity
from dam.models.core.base_component import BaseComponent
from dam.models.conceptual import (
    TagConceptComponent,
    EntityTagLinkComponent,
    ComicBookConceptComponent,
    ComicBookVariantComponent,
    BaseConceptualInfoComponent,
    BaseVariantInfoComponent
)
from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
import inspect

from dam.services import ecs_service

logger = logging.getLogger(__name__)

# --- Tag Definition Functions ---

def create_tag_concept(
    session: Session,
    tag_name: str,
    scope_type: str,
    scope_detail: Optional[str] = None,
    description: Optional[str] = None,
    allow_values: bool = False
) -> Optional[Entity]:
    if not tag_name:
        raise ValueError("Tag name cannot be empty.")
    if not scope_type:
        raise ValueError("Tag scope type cannot be empty.")

    existing_tag_concept = get_tag_concept_by_name(session, tag_name)
    if existing_tag_concept:
        logger.warning(f"TagConcept with name '{tag_name}' already exists with Entity ID {existing_tag_concept.id}.")
        return existing_tag_concept

    tag_concept_entity = ecs_service.create_entity(session)
    tag_concept_comp = TagConceptComponent(
        entity=tag_concept_entity,
        tag_name=tag_name,
        tag_scope_type=scope_type.upper(),
        tag_scope_detail=scope_detail,
        tag_description=description,
        allow_values=allow_values
    )
    session.add(tag_concept_comp)
    try:
        session.flush()
        logger.info(f"Created TagConcept Entity ID {tag_concept_entity.id} with name '{tag_name}'.")
        return tag_concept_entity
    except IntegrityError:
        session.rollback()
        logger.error(f"Failed to create TagConcept '{tag_name}' due to unique constraint violation (name likely exists).")
        return None

def get_tag_concept_by_name(session: Session, name: str) -> Optional[Entity]:
    stmt = (
        select(Entity)
        .join(TagConceptComponent, Entity.id == TagConceptComponent.entity_id)
        .where(TagConceptComponent.tag_name == name)
    )
    return session.execute(stmt).scalar_one_or_none()

def get_tag_concept_by_id(session: Session, tag_concept_entity_id: int) -> Optional[Entity]:
    tag_concept_entity = ecs_service.get_entity(session, tag_concept_entity_id)
    if tag_concept_entity and ecs_service.get_component(session, tag_concept_entity_id, TagConceptComponent):
        return tag_concept_entity
    return None

def find_tag_concepts(
    session: Session,
    query_name: Optional[str] = None,
    scope_type: Optional[str] = None
) -> List[Entity]:
    stmt = select(Entity).join(TagConceptComponent, Entity.id == TagConceptComponent.entity_id)
    if query_name:
        stmt = stmt.where(TagConceptComponent.tag_name.ilike(f"%{query_name}%"))
    if scope_type:
        stmt = stmt.where(TagConceptComponent.tag_scope_type == scope_type.upper())

    stmt = stmt.order_by(TagConceptComponent.tag_name)
    return session.execute(stmt).scalars().all()


def update_tag_concept(
    session: Session,
    tag_concept_entity_id: int,
    name: Optional[str] = None,
    scope_type: Optional[str] = None,
    scope_detail: Optional[str] = None,
    description: Optional[str] = None,
    allow_values: Optional[bool] = None
) -> Optional[TagConceptComponent]:
    tag_concept_comp = ecs_service.get_component(session, tag_concept_entity_id, TagConceptComponent)
    if not tag_concept_comp:
        logger.warning(f"TagConceptComponent not found for Entity ID {tag_concept_entity_id}.")
        return None

    updated = False
    if name is not None and tag_concept_comp.tag_name != name:
        existing_tag = get_tag_concept_by_name(session, name)
        if existing_tag and existing_tag.id != tag_concept_entity_id:
            logger.error(f"Cannot update tag name to '{name}' as it already exists for TagConcept ID {existing_tag.id}.")
            return None
        tag_concept_comp.tag_name = name
        updated = True
    if scope_type is not None and tag_concept_comp.tag_scope_type != scope_type.upper():
        tag_concept_comp.tag_scope_type = scope_type.upper()
        updated = True

    if scope_detail == "__CLEAR__":
        if tag_concept_comp.tag_scope_detail is not None:
            tag_concept_comp.tag_scope_detail = None
            updated = True
    elif scope_detail is not None and tag_concept_comp.tag_scope_detail != scope_detail:
        tag_concept_comp.tag_scope_detail = scope_detail
        updated = True

    if description == "__CLEAR__":
        if tag_concept_comp.tag_description is not None:
            tag_concept_comp.tag_description = None
            updated = True
    elif description is not None and tag_concept_comp.tag_description != description:
        tag_concept_comp.tag_description = description
        updated = True

    if allow_values is not None and tag_concept_comp.allow_values != allow_values:
        tag_concept_comp.allow_values = allow_values
        updated = True

    if updated:
        try:
            session.add(tag_concept_comp)
            session.flush()
            logger.info(f"Updated TagConceptComponent for Entity ID {tag_concept_entity_id}.")
        except IntegrityError:
            session.rollback()
            logger.error(f"Failed to update TagConcept '{tag_concept_comp.tag_name}' due to unique constraint violation (name likely exists).")
            return None
    return tag_concept_comp


def delete_tag_concept(session: Session, tag_concept_entity_id: int) -> bool:
    tag_concept_entity = get_tag_concept_by_id(session, tag_concept_entity_id)
    if not tag_concept_entity:
        logger.warning(f"TagConcept Entity ID {tag_concept_entity_id} not found for deletion.")
        return False

    stmt = delete(EntityTagLinkComponent).where(EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id)
    session.execute(stmt)

    return ecs_service.delete_entity(session, tag_concept_entity_id)


# --- Tag Application Functions ---

def _is_scope_valid(
    session: Session,
    entity_id_to_tag: int,
    tag_concept_comp: TagConceptComponent
) -> bool:
    # print(f"DEBUG: _is_scope_valid called for entity {entity_id_to_tag}, tag '{tag_concept_comp.tag_name}'")
    # print(f"DEBUG: REGISTERED_COMPONENT_TYPES at start of _is_scope_valid: {[c.__name__ for c in REGISTERED_COMPONENT_TYPES]}")

    scope_type = tag_concept_comp.tag_scope_type
    scope_detail = tag_concept_comp.tag_scope_detail

    if scope_type == "GLOBAL":
        return True

    if scope_type == "COMPONENT_CLASS_REQUIRED":
        if not scope_detail:
            logger.error(f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_comp.entity_id}) has scope COMPONENT_CLASS_REQUIRED but no scope_detail (component class name).")
            return False

        required_class: Optional[Type[BaseComponent]] = None
        for comp_class in REGISTERED_COMPONENT_TYPES:
            if comp_class.__name__ == scope_detail:
                required_class = comp_class
                break

        if not required_class:
            logger.error(f"Scope validation failed: Component class '{scope_detail}' for tag '{tag_concept_comp.tag_name}' is not a registered component type.")
            return False

        if not ecs_service.get_component(session, entity_id_to_tag, required_class):
            logger.warning(f"Scope validation failed: Entity {entity_id_to_tag} does not have required component '{scope_detail}' for tag '{tag_concept_comp.tag_name}'.")
            return False
        return True

    if scope_type == "CONCEPTUAL_ASSET_LOCAL":
        if not scope_detail:
            logger.error(f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_comp.entity_id}) has scope CONCEPTUAL_ASSET_LOCAL but no scope_detail (conceptual asset entity ID).")
            return False
        try:
            conceptual_asset_entity_id_for_scope = int(scope_detail)
        except ValueError:
            logger.error(f"Invalid scope_detail '{scope_detail}' for CONCEPTUAL_ASSET_LOCAL scope of tag '{tag_concept_comp.tag_name}'. Expected integer Entity ID.")
            return False

        scope_owner_is_conceptual_asset = False
        if ecs_service.get_component(session, conceptual_asset_entity_id_for_scope, ComicBookConceptComponent):
            scope_owner_is_conceptual_asset = True
        else:
            for comp_type_check in REGISTERED_COMPONENT_TYPES:
                if inspect.isclass(comp_type_check) and \
                   issubclass(comp_type_check, BaseConceptualInfoComponent) and \
                   not comp_type_check.__dict__.get('__abstract__', False): # Use __dict__.get for direct check
                    if ecs_service.get_component(session, conceptual_asset_entity_id_for_scope, comp_type_check):
                        scope_owner_is_conceptual_asset = True
                        break

        if not scope_owner_is_conceptual_asset:
            logger.error(f"Scope detail ID {conceptual_asset_entity_id_for_scope} for tag '{tag_concept_comp.tag_name}' (scope type CONCEPTUAL_ASSET_LOCAL) does not refer to a valid conceptual asset entity.")
            return False

        if entity_id_to_tag == conceptual_asset_entity_id_for_scope:
            return True
        else:
            is_valid_variant = False
            for comp_type_check in REGISTERED_COMPONENT_TYPES:
                is_class = inspect.isclass(comp_type_check)
                is_variant_subclass = issubclass(comp_type_check, BaseVariantInfoComponent) if is_class else False
                # Correctly check if the class itself is concrete (not inheriting __abstract__ from parent)
                is_actually_concrete = not comp_type_check.__dict__.get('__abstract__', False) if is_class else False

                if is_class and is_variant_subclass and is_actually_concrete:
                    variant_comp = ecs_service.get_component(session, entity_id_to_tag, comp_type_check)
                    if variant_comp:
                        if hasattr(variant_comp, 'conceptual_entity_id'):
                            actual_conceptual_id = variant_comp.conceptual_entity_id
                            if actual_conceptual_id == conceptual_asset_entity_id_for_scope:
                                is_valid_variant = True
                                break

            if is_valid_variant:
                return True

        logger.warning(f"Scope validation failed: Entity {entity_id_to_tag} is not the specified conceptual asset {conceptual_asset_entity_id_for_scope} nor its variant for tag '{tag_concept_comp.tag_name}'.")
        return False

    logger.warning(f"Unknown tag_scope_type '{scope_type}' for tag '{tag_concept_comp.tag_name}'. Denying application by default for unknown scopes.")
    return False


def apply_tag_to_entity(
    session: Session,
    entity_id_to_tag: int,
    tag_concept_entity_id: int,
    value: Optional[str] = None
) -> Optional[EntityTagLinkComponent]:
    target_entity = ecs_service.get_entity(session, entity_id_to_tag)
    if not target_entity:
        logger.error(f"Entity to tag (ID: {entity_id_to_tag}) not found.")
        return None

    tag_concept_entity = get_tag_concept_by_id(session, tag_concept_entity_id)
    if not tag_concept_entity:
        logger.error(f"TagConcept Entity (ID: {tag_concept_entity_id}) not found.")
        return None

    tag_concept_comp = ecs_service.get_component(session, tag_concept_entity_id, TagConceptComponent)
    if not tag_concept_comp:
        logger.error(f"TagConceptComponent missing on Entity ID {tag_concept_entity_id}.")
        return None

    if not _is_scope_valid(session, entity_id_to_tag, tag_concept_comp):
        return None

    if not tag_concept_comp.allow_values and value is not None:
        logger.warning(f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_entity_id}) does not allow values, but value '{value}' provided. Value will be ignored.")
        value = None

    existing_link_stmt = select(EntityTagLinkComponent).where(
        EntityTagLinkComponent.entity_id == entity_id_to_tag,
        EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id,
        EntityTagLinkComponent.tag_value == value
    )
    existing_link = session.execute(existing_link_stmt).scalar_one_or_none()

    if existing_link:
         logger.warning(f"Tag '{tag_concept_comp.tag_name}' with value '{value}' already applied to Entity {entity_id_to_tag}. Not applying again.")
         return None

    link_comp = EntityTagLinkComponent(
        entity=target_entity,
        tag_concept=tag_concept_entity,
        tag_value=value
    )

    try:
        session.add(link_comp)
        session.flush()
        logger.info(f"Applied tag '{tag_concept_comp.tag_name}' (Concept ID: {tag_concept_entity_id}) to Entity ID {entity_id_to_tag} with value '{value}'.")
        return link_comp
    except IntegrityError:
        session.rollback()
        logger.error(f"Failed to apply tag '{tag_concept_comp.tag_name}' to Entity {entity_id_to_tag} (value: '{value}'). Likely duplicate application (this should have been caught by pre-check).")
        return None
    except Exception as e:
        session.rollback()
        logger.error(f"An unexpected error occurred while applying tag: {e}")
        raise


def remove_tag_from_entity(
    session: Session,
    entity_id_tagged: int,
    tag_concept_entity_id: int,
    value: Optional[str] = None
) -> bool:
    stmt = select(EntityTagLinkComponent).where(
        EntityTagLinkComponent.entity_id == entity_id_tagged,
        EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id,
        EntityTagLinkComponent.tag_value == value
    )
    link_comp_to_delete = session.execute(stmt).scalar_one_or_none()

    if link_comp_to_delete:
        session.delete(link_comp_to_delete)
        logger.info(f"Removed tag (Concept ID: {tag_concept_entity_id}, Value: '{value}') from Entity ID {entity_id_tagged}.")
        return True
    logger.warning(f"Tag application (Concept ID: {tag_concept_entity_id}, Value: '{value}') not found on Entity ID {entity_id_tagged}.")
    return False


def get_tags_for_entity(session: Session, entity_id_tagged: int) -> List[Tuple[Entity, Optional[str]]]:
    stmt = (
        select(EntityTagLinkComponent.tag_concept_entity_id, EntityTagLinkComponent.tag_value)
        .where(EntityTagLinkComponent.entity_id == entity_id_tagged)
    )
    results = session.execute(stmt).all()

    tags_info = []
    for tag_concept_id, tag_val in results:
        tag_concept_e = get_tag_concept_by_id(session, tag_concept_id)
        if tag_concept_e:
            tags_info.append((tag_concept_e, tag_val))
    return tags_info


def get_entities_for_tag(
    session: Session,
    tag_concept_entity_id: int,
    value_filter: Optional[str] = None,
    filter_by_value_presence: Optional[bool] = None
) -> List[Entity]:
    stmt = (
        select(Entity)
        .join(EntityTagLinkComponent, Entity.id == EntityTagLinkComponent.entity_id)
        .where(EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id)
    )
    if value_filter is not None:
        stmt = stmt.where(EntityTagLinkComponent.tag_value == value_filter)
    elif filter_by_value_presence is True:
        stmt = stmt.where(EntityTagLinkComponent.tag_value.isnot(None))
    elif filter_by_value_presence is False:
        stmt = stmt.where(EntityTagLinkComponent.tag_value.is_(None))

    stmt = stmt.distinct()
    return session.execute(stmt).scalars().all()
