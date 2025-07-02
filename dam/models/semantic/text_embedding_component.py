from typing import Optional  # List for the embedding itself

from sqlalchemy import LargeBinary, String  # For embedding storage
from sqlalchemy.orm import Mapped, mapped_column

from ..core import BaseComponent


class TextEmbeddingComponent(BaseComponent):
    """
    Stores text embeddings for an entity, generated from one of its text fields.
    """

    __tablename__ = "component_text_embedding"

    # The actual embedding vector.
    # Storing as LargeBinary is DB-agnostic for the raw bytes of the vector.
    # For PostgreSQL, ARRAY(Float) might be more natural for querying with pgvector.
    # For simplicity and broader compatibility first, LargeBinary.
    # Consider a fixed size based on the model used, e.g., 384 floats * 4 bytes/float = 1536 bytes for 'all-MiniLM-L6-v2'
    # This stores a numpy array (e.g., np.ndarray of dtype float32, shape (384,)) converted to bytes.
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # Store as bytes

    # Name of the sentence transformer model used to generate this embedding.
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Information about the source of the text that was embedded.
    # This helps in understanding what this embedding represents.
    source_component_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_field_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Could also store original text for inspection, but might be redundant if source component is versioned.
    # source_text_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Example for PostgreSQL specific type (requires pgvector or similar for useful indexing/search)
    # embedding_vector_pg: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)

    def __repr__(self):
        return (
            f"TextEmbeddingComponent(id={self.id}, entity_id={self.entity_id}, "
            f"model_name='{self.model_name}', source='{self.source_component_name}.{self.source_field_name}', "
            f"embedding_vector_len={len(self.embedding_vector) if self.embedding_vector else 0} bytes)"
        )

    # Note: For actual use, helper methods to convert list <-> bytes are crucial.
    # For now, we'll assume the service layer handles this conversion before saving
    # and after retrieving. Adding numpy as a dependency might be the easiest way.
    # If not, `struct` module can be used.
