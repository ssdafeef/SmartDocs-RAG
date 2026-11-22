"""Unit tests for the FAISS vector store and metadata management subsystem."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.rag.models import ExtractedDocument, PageContent, TextChunk
from backend.rag.vector_store import FAISSVectorStore
from backend.utils.exceptions import DocumentNotFoundError


@pytest.fixture
def temp_vector_store():
    """Create a vector store fixture using isolated temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        index_file = tmp_path / "test.index"
        docstore_file = tmp_path / "test_meta.json"
        store = FAISSVectorStore(dimension=4, index_path=index_file, docstore_path=docstore_file)
        yield store


def test_add_and_search_vector_store(temp_vector_store):
    """Verify vectors can be indexed and searched by cosine similarity."""
    store = temp_vector_store

    doc = ExtractedDocument(
        document_id="doc-1",
        filename="test_rag.pdf",
        file_path="/tmp/test_rag.pdf",
        file_hash="hash1",
        total_pages=1,
        pages=[PageContent(page_number=1, text="Sample text")],
    )

    chunks = [
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="test_rag.pdf",
            page_number=1,
            chunk_index=0,
            text="Artificial intelligence and RAG.",
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            filename="test_rag.pdf",
            page_number=1,
            chunk_index=1,
            text="Baking chocolate cake recipes.",
        ),
    ]

    # Two normalized 4-dimensional vectors
    # Vector 1 points along axis 0
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Vector 2 points along axis 1
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.stack([v1, v2])

    store.add_document_chunks(document=doc, chunks=chunks, embeddings=embeddings)

    assert store.index.ntotal == 2
    assert len(store.chunks) == 2
    assert "doc-1" in store.documents

    # Search with a query identical to vector 1
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=2, similarity_threshold=0.5)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert abs(results[0].score - 1.0) < 1e-4


def test_delete_document_nonexistent(temp_vector_store):
    """Verify DocumentNotFoundError is raised when deleting unknown ID."""
    with pytest.raises(DocumentNotFoundError):
        temp_vector_store.delete_document("nonexistent-id-999")


def test_clear_all(temp_vector_store):
    """Verify clear_all resets index and removes documents."""
    store = temp_vector_store
    doc = ExtractedDocument(
        document_id="doc-clear",
        filename="clear.pdf",
        file_path="/tmp/clear.pdf",
        file_hash="hash2",
        total_pages=1,
        pages=[PageContent(page_number=1, text="Sample")],
    )
    store.add_document_chunks(
        document=doc,
        chunks=[],
        embeddings=np.empty((0, 4), dtype=np.float32),
    )

    assert len(store.documents) == 1
    store.clear_all()
    assert len(store.documents) == 0
    assert store.index.ntotal == 0
