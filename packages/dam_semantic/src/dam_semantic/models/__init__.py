# Old component, to be removed once service layer is updated and migrations are handled
# Now importing the renamed OldTextEmbeddingComponent directly
from .text_embedding_component import (
    EMBEDDING_MODEL_REGISTRY,
    BaseSpecificEmbeddingComponent,
    EmbeddingModelInfo,  # Optional: if used by other modules
    ModelHyperparameters,  # Optional: if used by other modules
    OldTextEmbeddingComponent,
    TextEmbeddingAllMiniLML6V2Dim384Component,
    TextEmbeddingClipVitB32Dim512Component,
    get_embedding_component_class,
)

__all__ = [
    "AUDIO_EMBEDDING_MODEL_REGISTRY",
    "EMBEDDING_MODEL_REGISTRY",
    "AudioEmbeddingModelInfo",
    "AudioEmbeddingPannsCnn14Dim2048Component",
    "AudioEmbeddingVggishDim128Component",
    "AudioModelHyperparameters",
    # Audio Embedding Components
    "BaseSpecificAudioEmbeddingComponent",
    "BaseSpecificEmbeddingComponent",
    "EmbeddingModelInfo",
    "ModelHyperparameters",
    "OldTextEmbeddingComponent",  # Keep for now for transition
    "TextEmbeddingAllMiniLML6V2Dim384Component",
    "TextEmbeddingClipVitB32Dim512Component",
    "get_audio_embedding_component_class",
    "get_embedding_component_class",
]

from .audio_embedding_component import (
    AUDIO_EMBEDDING_MODEL_REGISTRY,
    AudioEmbeddingModelInfo,  # Optional: if used by other modules
    AudioEmbeddingPannsCnn14Dim2048Component,
    AudioEmbeddingVggishDim128Component,
    AudioModelHyperparameters,  # Optional: if used by other modules
    BaseSpecificAudioEmbeddingComponent,
    get_audio_embedding_component_class,
)
