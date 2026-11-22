"""Vector Embeddings Generation Module for SmartDocs.

Generates dense vector representations using Sentence Transformers (all-MiniLM-L6-v2)
with L2 normalization for accurate cosine similarity calculation in FAISS.
"""

from typing import List, Optional, Union
import numpy as np

from backend.config import settings
from backend.rag.models import TextChunk
from backend.utils.exceptions import EmbeddingGenerationError
from backend.utils.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingManager:
    """Singleton manager for the Sentence Transformer embedding model."""

    _instance: Optional["EmbeddingManager"] = None
    _model = None

    def __new__(cls) -> "EmbeddingManager":
        """Ensure only one instance of EmbeddingManager exists in the process."""
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls):
        """Lazy-load the SentenceTransformer model into memory."""
        if cls._model is None:
            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'...",
                settings.EMBEDDING_MODEL_NAME,
                settings.EMBEDDING_DEVICE,
            )
            try:
                from sentence_transformers import SentenceTransformer

                # Try loading from local HuggingFace cache first to avoid network delays
                try:
                    cls._model = SentenceTransformer(
                        model_name_or_path=settings.EMBEDDING_MODEL_NAME,
                        device=settings.EMBEDDING_DEVICE,
                        local_files_only=True,
                    )
                except Exception:
                    cls._model = SentenceTransformer(
                        model_name_or_path=settings.EMBEDDING_MODEL_NAME,
                        device=settings.EMBEDDING_DEVICE,
                        local_files_only=False,
                    )
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as exc:
                logger.error("Failed to load SentenceTransformer model: %s", exc)
                raise EmbeddingGenerationError(
                    f"Could not load embedding model '{settings.EMBEDDING_MODEL_NAME}': {exc}"
                ) from exc
        return cls._model

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate normalized vector embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.
            normalize: Whether to L2-normalize vectors (vital for cosine similarity).

        Returns:
            2D numpy array of shape (N, dimension) with dtype float32.

        Raises:
            EmbeddingGenerationError: If encoding fails or produces invalid dimensions.
        """
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)

        model = self.get_model()

        try:
            # sentence-transformers 6.x renamed 'sentences' to 'inputs'
            try:
                embeddings = model.encode(
                    inputs=texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )
            except TypeError:
                embeddings = model.encode(
                    sentences=texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )

            # Ensure 32-bit floating point for FAISS compatibility
            embeddings_f32 = np.asarray(embeddings, dtype=np.float32)

            if embeddings_f32.ndim == 1:
                embeddings_f32 = np.expand_dims(embeddings_f32, axis=0)

            if embeddings_f32.shape[1] != settings.EMBEDDING_DIMENSION:
                raise EmbeddingGenerationError(
                    f"Expected embedding dimension {settings.EMBEDDING_DIMENSION}, "
                    f"got {embeddings_f32.shape[1]}."
                )

            return embeddings_f32

        except Exception as exc:
            logger.error("Error during embedding generation: %s", exc)
            raise EmbeddingGenerationError(f"Embedding encoding failed: {exc}") from exc

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single 1D normalized embedding vector for a search query.

        Args:
            query: User's question or search query string.

        Returns:
            1D numpy array of shape (dimension,) with dtype float32.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise EmbeddingGenerationError("Cannot embed an empty query string.")

        embeddings = self.embed_texts([cleaned_query], normalize=True)
        return embeddings[0]

    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract text from a list of TextChunk objects and generate embeddings.

        Args:
            chunks: List of TextChunk models.
            batch_size: Inference batch size.

        Returns:
            2D numpy array of shape (len(chunks), dimension).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts=texts, batch_size=batch_size, normalize=True)


# Global singleton instance
embedding_manager = EmbeddingManager()
