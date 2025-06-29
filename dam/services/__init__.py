# This file makes the 'services' directory a Python package.

# Import functions from the new tag service
from . import (
    ecs_service,
    file_operations,
    tag_service,  # noqa: F401
    transcode_service,
    world_service,
    character_service,
    semantic_service, # Added semantic_service
)

# Import functions from the new comic book service
from .comic_book_service import (
    create_comic_book_concept,
    find_comic_book_concepts,
    get_comic_concept_for_variant,
    get_primary_variant_for_comic_concept,
    get_variants_for_comic_concept,
    link_comic_variant_to_concept,
    set_primary_comic_variant,
    unlink_comic_variant,
)
from .ecs_service import (
    add_component_to_entity,
    create_entity,
    delete_entity,
    find_entities_by_component_attribute_value,
    find_entities_with_components,  # Exposing more ecs_service functions
    find_entity_by_content_hash,
    get_component,
    get_components,
    get_components_by_value,
    get_entity,
    remove_component,
)
from .tag_service import (
    apply_tag_to_entity,
    create_tag_concept,
    delete_tag_concept,
    find_tag_concepts,
    get_entities_for_tag,
    get_tag_concept_by_id,
    get_tag_concept_by_name,
    get_tags_for_entity,
    remove_tag_from_entity,
    update_tag_concept,
)

# Import functions from the new transcode service
from .transcode_service import (
    TranscodeServiceError,
    apply_transcode_profile,
    create_transcode_profile,
    get_assets_using_profile,
    get_transcode_profile_by_name_or_id,
    get_transcoded_variants_for_original,
)

# Import functions from the new character service
from .character_service import (
    CharacterConceptNotFoundError, # Exceptions
    CharacterLinkNotFoundError,
    create_character_concept,
    get_character_concept_by_name,
    get_character_concept_by_id,
    find_character_concepts,
    update_character_concept,
    delete_character_concept,
    apply_character_to_entity,
    remove_character_from_entity,
    get_characters_for_entity,
    get_entities_for_character,
)

# Import functions from the new semantic service
from .semantic_service import (
    generate_embedding,
    convert_embedding_to_bytes,
    convert_bytes_to_embedding,
    update_text_embeddings_for_entity,
    get_text_embeddings_for_entity,
    find_similar_entities_by_text_embedding,
    get_sentence_transformer_model, # Expose if direct model access is needed elsewhere
    DEFAULT_MODEL_NAME, # Expose default model name for consistency
)

__all__ = [
    "file_operations",
    "ecs_service",  # Module itself
    "world_service",  # Module itself
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
    "tag_service",  # Module itself
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
    "transcode_service",  # Module itself
    "create_transcode_profile",
    "get_transcode_profile_by_name_or_id",
    "apply_transcode_profile",
    "get_transcoded_variants_for_original",
    "get_assets_using_profile",
    "TranscodeServiceError",
    # Character service and functions
    "character_service", # Module itself
    "create_character_concept",
    "get_character_concept_by_name",
    "get_character_concept_by_id",
    "find_character_concepts",
    "update_character_concept",
    "delete_character_concept",
    "apply_character_to_entity",
    "remove_character_from_entity",
    "get_characters_for_entity",
    "get_entities_for_character",
    # Semantic service and functions
    "semantic_service", # Module itself
    "generate_embedding",
    "convert_embedding_to_bytes",
    "convert_bytes_to_embedding",
    "update_text_embeddings_for_entity",
    "get_text_embeddings_for_entity",
    "find_similar_entities_by_text_embedding",
    "get_sentence_transformer_model",
    "DEFAULT_MODEL_NAME",
]
