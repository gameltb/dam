from .text_embedding_component import (
    BaseSpecificEmbeddingComponent,
    TextEmbeddingAllMiniLML6V2Dim384Component,
    TextEmbeddingClipVitB32Dim512Component,
    EMBEDDING_MODEL_REGISTRY,
    get_embedding_component_class,
    ModelHyperparameters, # Optional: if used by other modules
    EmbeddingModelInfo, # Optional: if used by other modules
)

# Old component, to be removed once service layer is updated and migrations are handled
# Now importing the renamed OldTextEmbeddingComponent directly
from .text_embedding_component import OldTextEmbeddingComponent

__all__ = [
    "BaseSpecificEmbeddingComponent",
    "TextEmbeddingAllMiniLML6V2Dim384Component",
    "TextEmbeddingClipVitB32Dim512Component",
    "EMBEDDING_MODEL_REGISTRY",
    "get_embedding_component_class",
    "ModelHyperparameters",
    "EmbeddingModelInfo",
    "OldTextEmbeddingComponent", # Keep for now for transition
]
