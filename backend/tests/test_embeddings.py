"""Unit tests for the Sentence Transformer embedding generation module."""

import numpy as np
import pytest

from backend.rag.embeddings import embedding_manager
from backend.rag.models import TextChunk


def test_embed_query_dimension_and_norm():
    """Verify embed_query outputs 384-dimensional unit vector."""
    query = "What is Retrieval-Augmented Generation?"
    vec = embedding_manager.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # L2 norm of normalized vector must equal 1.0
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_embed_texts_empty():
    """Verify embed_texts returns empty array when given no texts."""
    empty_vecs = embedding_manager.embed_texts([])
    assert empty_vecs.shape == (0, 384)


def test_embed_chunks():
    """Verify embedding generation over TextChunk models."""
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            text="Artificial intelligence enables computer systems to perform human tasks.",
        ),
        TextChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            filename="test.pdf",
            page_number=1,
            chunk_index=1,
            text="Machine learning algorithms build models based on sample training data.",
        ),
    ]

    embeddings = embedding_manager.embed_chunks(chunks)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_semantic_similarity_relative_ordering():
    """Verify that semantically related phrases have higher dot products than unrelated phrases."""
    v_ai = embedding_manager.embed_query("artificial intelligence and machine learning")
    v_dl = embedding_manager.embed_query("deep neural networks and backpropagation")
    v_cooking = embedding_manager.embed_query("chocolate cake baking recipe and frosting")

    sim_related = np.dot(v_ai, v_dl)
    sim_unrelated = np.dot(v_ai, v_cooking)

    assert sim_related > sim_unrelated
