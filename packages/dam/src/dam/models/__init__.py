# This file makes the 'models' directory a Python package.

# Ensure all model modules are imported so they register with Base.metadata

# Conceptual components
# Import the concrete components that are intended for direct use/instantiation
from .conceptual.comic_book_concept_component import ComicBookConceptComponent
from .conceptual.comic_book_variant_component import ComicBookVariantComponent

# EntityTagLinkComponent and TagConceptComponent moved to .tags
from .conceptual.page_link import PageLink

# Core model elements
from .core.base_class import Base
from .core.base_component import BaseComponent
from .core.entity import Entity

# Hash components
from .hashes.content_hash_md5_component import ContentHashMD5Component
from .hashes.content_hash_sha256_component import ContentHashSHA256Component

# Metadata components
from .metadata.exiftool_metadata_component import ExiftoolMetadataComponent

# Property components
# Semantic components
# Source info components
from .tags.entity_tag_link_component import EntityTagLinkComponent
from .tags.model_generated_tag_link_component import ModelGeneratedTagLinkComponent

# Tag related components (now in their own package)
from .tags.tag_concept_component import TagConceptComponent

# Base abstract conceptual components like BaseConceptualInfoComponent and BaseVariantInfoComponent
# are typically not re-exported here unless specifically needed for widespread type hinting.
# Their subclasses (the concrete components above) are what services and systems will primarily work with.

__all__ = [
    # Core
    "Base",
    "Entity",
    "BaseComponent",
    # Hashes
    "ContentHashMD5Component",
    "ContentHashSHA256Component",
    # Properties
    # Metadata
    "ExiftoolMetadataComponent",
    # Source Info
    # Conceptual (Concrete Components and Association Objects)
    "ComicBookConceptComponent",
    "ComicBookVariantComponent",
    "PageLink",
    # Conceptual (Tag related components are now under their own section)
    # Tags
    "TagConceptComponent",
    "EntityTagLinkComponent",
    "ModelGeneratedTagLinkComponent",
]
