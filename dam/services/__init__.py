# This file makes the 'services' directory a Python package.

from . import ecs_service, file_operations, world_service
from .ecs_service import (
    add_component_to_entity,
    create_entity,
    delete_entity,
    get_component,
    get_components,
    get_entity,
    remove_component,
    find_entities_with_components, # Exposing more ecs_service functions
    get_components_by_value,
    find_entity_by_content_hash,
    find_entities_by_component_attribute_value,
)

# Import functions from the new comic book service
from .comic_book_service import (
    create_comic_book_concept,
    link_comic_variant_to_concept,
    get_variants_for_comic_concept,
    get_comic_concept_for_variant,
    find_comic_book_concepts,
    set_primary_comic_variant,
    get_primary_variant_for_comic_concept,
    unlink_comic_variant,
)

__all__ = [
    "file_operations",
    "ecs_service", # Module itself
    "world_service", # Module itself
    # Functions from ecs_service
    "add_component_to_entity",
    "create_entity",
    "delete_entity",
    "get_entity",
    "get_component",
    "get_components",
    "remove_component",
    "find_entities_with_components",
    "get_components_by_value",
    "find_entity_by_content_hash",
    "find_entities_by_component_attribute_value",
    # Functions from comic_book_service
    "create_comic_book_concept",
    "link_comic_variant_to_concept",
    "get_variants_for_comic_concept",
    "get_comic_concept_for_variant",
    "find_comic_book_concepts",
    "set_primary_comic_variant",
    "get_primary_variant_for_comic_concept",
    "unlink_comic_variant",
    # Tagging service and functions will be added here
]

# Import functions from the new tag service
from . import tag_service # Import the module first
from .tag_service import (
    create_tag_concept,
    get_tag_concept_by_name,
    get_tag_concept_by_id,
    find_tag_concepts,
    update_tag_concept,
    delete_tag_concept,
    apply_tag_to_entity,
    remove_tag_from_entity,
    get_tags_for_entity,
    get_entities_for_tag,
)

# Add new service module and its functions to __all__
__all__.extend([
    "tag_service", # Module itself
    "create_tag_concept",
    "get_tag_concept_by_name",
    "get_tag_concept_by_id",
    "find_tag_concepts",
    "update_tag_concept",
    "delete_tag_concept",
    "apply_tag_to_entity",
    "remove_tag_from_entity",
    "get_tags_for_entity",
    "get_entities_for_tag",
])
