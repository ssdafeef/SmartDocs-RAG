"""FAISS Vector Store and Metadata Management Module for SmartDocs.

Stores dense vector embeddings with FAISS IndexFlatIP (cosine similarity)
and maintains synchronized chunk metadata with disk persistence and full CRUD operations.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from backend.config import settings
from backend.rag.models import ExtractedDocument, SearchResult, TextChunk
from backend.utils.exceptions import DocumentNotFoundError, VectorStoreError
from backend.utils.logger import get_logger

logger = get_logger("vector_store")


class FAISSVectorStore:
    """Manages FAISS vector index and document metadata storage."""

    def __init__(
        self,
        dimension: int = settings.EMBEDDING_DIMENSION,
        index_path: Optional[Path] = None,
        docstore_path: Optional[Path] = None,
    ) -> None:
        """Initialize FAISS index and metadata store.

        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2).
            index_path: Custom path to save/load FAISS index.
            docstore_path: Custom path to save/load metadata JSON.
        """
        self.dimension = dimension
        self.index_path = index_path or settings.FAISS_INDEX_FILE
        self.docstore_path = docstore_path or settings.DOCSTORE_FILE

        self._lock = threading.RLock()
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[TextChunk] = []
        self.documents: Dict[str, Dict[str, Any]] = {}

        # Attempt to load persistent state if available
        self.load()

    def add_document_chunks(
        self,
        document: ExtractedDocument,
        chunks: List[TextChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add a document and its text chunks with embeddings to the vector store.

        Args:
            document: ExtractedDocument model.
            chunks: List of TextChunk models.
            embeddings: 2D numpy array of shape (len(chunks), dimension).

        Raises:
            VectorStoreError: If vector shapes do not match chunk counts.
        """
        with self._lock:
            if len(chunks) == 0:
                logger.warning("No chunks provided for document '%s'.", document.filename)
                self.documents[document.document_id] = {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "file_path": document.file_path,
                    "file_hash": document.file_hash,
                    "file_size_bytes": document.file_size_bytes,
                    "total_pages": document.total_pages,
                    "total_chars": document.total_chars,
                    "chunk_count": 0,
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                    "metadata": document.metadata,
                }
                self.save()
                return

            if embeddings.shape[0] != len(chunks):
                raise VectorStoreError(
                    f"Embeddings count ({embeddings.shape[0]}) does not match "
                    f"chunks count ({len(chunks)})."
                )

            if embeddings.shape[1] != self.dimension:
                raise VectorStoreError(
                    f"Embedding dimension ({embeddings.shape[1]}) does not match "
                    f"index dimension ({self.dimension})."
                )

            # Ensure 32-bit float
            emb_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

            # Add vectors to FAISS index
            self.index.add(emb_f32)
            self.chunks.extend(chunks)

            # Record document metadata
            self.documents[document.document_id] = {
                "document_id": document.document_id,
                "filename": document.filename,
                "file_path": document.file_path,
                "file_hash": document.file_hash,
                "file_size_bytes": document.file_size_bytes,
                "total_pages": document.total_pages,
                "total_chars": document.total_chars,
                "chunk_count": len(chunks),
                "upload_date": datetime.now(timezone.utc).isoformat(),
                "metadata": document.metadata,
            }

            logger.info(
                "Indexed document '%s' with %d chunks. Total vectors in index: %d",
                document.filename,
                len(chunks),
                self.index.ntotal,
            )

            # Persist changes to disk
            self.save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Perform semantic similarity search using cosine similarity (inner product).

        Args:
            query_vector: 1D or 2D numpy array of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            similarity_threshold: Minimum similarity score required.

        Returns:
            List of SearchResult objects sorted by descending similarity score.
        """
        with self._lock:
            k = top_k or settings.TOP_K_RETRIEVAL
            threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

            if self.index.ntotal == 0 or len(self.chunks) == 0:
                logger.info("Search requested but vector index is empty.")
                return []

            # Format query vector
            q_vec = np.asarray(query_vector, dtype=np.float32)
            if q_vec.ndim == 1:
                q_vec = np.expand_dims(q_vec, axis=0)
            q_vec = np.ascontiguousarray(q_vec)

            # Query cannot request more vectors than exist in the index
            effective_k = min(k, self.index.ntotal)

            scores, indices = self.index.search(q_vec, effective_k)

            results: List[SearchResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue

                sim_score = float(score)
                if sim_score >= threshold:
                    results.append(
                        SearchResult(
                            chunk=self.chunks[idx],
                            score=sim_score,
                        )
                    )

            logger.info(
                "Search found %d matches above threshold %.2f (top_k=%d, total_in_index=%d)",
                len(results),
                threshold,
                k,
                self.index.ntotal,
            )

            return results

    def delete_document(self, document_id: str) -> bool:
        """Remove a document and rebuild the FAISS index with remaining chunks.

        Args:
            document_id: UUID of document to remove.

        Returns:
            True if document was found and removed.

        Raises:
            DocumentNotFoundError: If document_id does not exist.
        """
        with self._lock:
            if document_id not in self.documents:
                raise DocumentNotFoundError(document_id)

            doc_info = self.documents.pop(document_id)
            logger.info("Removing document '%s' (id: %s)", doc_info.get("filename"), document_id)

            # Filter out chunks belonging to this document
            remaining_chunks = [c for c in self.chunks if c.document_id != document_id]

            if len(remaining_chunks) == 0:
                # Reset to fresh empty index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks = []
            else:
                # Need to re-embed remaining chunks to rebuild the index accurately
                from backend.rag.embeddings import embedding_manager

                self.chunks = remaining_chunks
                embeddings = embedding_manager.embed_chunks(remaining_chunks)

                new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
                self.index = new_index

            # Delete source file from uploads dir if it exists
            file_path = doc_info.get("file_path")
            if file_path:
                try:
                    p = Path(file_path)
                    if p.exists() and p.is_file():
                        p.unlink()
                        logger.info("Deleted underlying file: %s", file_path)
                except Exception as exc:
                    logger.warning("Could not delete file %s: %s", file_path, exc)

            self.save()
            logger.info("Document removed successfully. Remaining vectors: %d", self.index.ntotal)
            return True

    def clear_all(self) -> None:
        """Wipe the entire vector store, metadata, and all uploaded files."""
        with self._lock:
            logger.warning("Clearing all indexed documents and vector storage.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self.documents = {}

            # Remove persistent files
            if self.index_path.exists():
                try:
                    self.index_path.unlink()
                except Exception as exc:
                    logger.warning("Failed to delete FAISS index file: %s", exc)

            if self.docstore_path.exists():
                try:
                    self.docstore_path.unlink()
                except Exception as exc:
                    logger.warning("Failed to delete metadata file: %s", exc)

            # Clean uploads directory
            uploads_dir = settings.UPLOADS_DIR
            if uploads_dir.exists():
                for item in uploads_dir.glob("*.pdf"):
                    try:
                        item.unlink()
                    except Exception as exc:
                        logger.warning("Failed to delete uploaded file %s: %s", item, exc)

            logger.info("Vector store reset complete.")

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return list of all currently indexed documents."""
        with self._lock:
            return list(self.documents.values())

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific document ID."""
        with self._lock:
            return self.documents.get(document_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for the vector database."""
        with self._lock:
            return {
                "total_documents": len(self.documents),
                "total_chunks": len(self.chunks),
                "total_vectors": self.index.ntotal,
                "embedding_dimension": self.dimension,
                "index_file_exists": self.index_path.exists(),
                "docstore_file_exists": self.docstore_path.exists(),
            }

    def save(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        with self._lock:
            try:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.docstore_path.parent.mkdir(parents=True, exist_ok=True)

                # Write FAISS binary index
                faiss.write_index(self.index, str(self.index_path))

                # Serialize metadata and chunks
                serialized_chunks = [chunk.model_dump() for chunk in self.chunks]
                payload = {
                    "documents": self.documents,
                    "chunks": serialized_chunks,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }

                with open(self.docstore_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

                logger.info(
                    "Persisted vector store (%d vectors, %d docs) to %s",
                    self.index.ntotal,
                    len(self.documents),
                    self.index_path.name,
                )
            except Exception as exc:
                logger.error("Failed to persist vector store: %s", exc)
                raise VectorStoreError(f"Persistence error: {exc}") from exc

    def load(self) -> bool:
        """Load FAISS index and chunk metadata from disk if available."""
        with self._lock:
            if not self.index_path.exists() or not self.docstore_path.exists():
                logger.info("No existing vector store files found on disk. Starting fresh.")
                return False

            try:
                # Load FAISS index
                loaded_index = faiss.read_index(str(self.index_path))
                self.index = loaded_index

                # Load metadata
                with open(self.docstore_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                self.documents = payload.get("documents", {})
                raw_chunks = payload.get("chunks", [])
                self.chunks = [TextChunk(**c) for c in raw_chunks]

                logger.info(
                    "Loaded vector store: %d vectors, %d documents from %s",
                    self.index.ntotal,
                    len(self.documents),
                    self.index_path.name,
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load existing vector store: %s. Starting fresh.", exc)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks = []
                self.documents = {}
                return False


# Global singleton vector store instance
vector_store = FAISSVectorStore()
