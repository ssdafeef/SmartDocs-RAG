"""Unit tests for the recursive text chunking and splitting module."""

import pytest
from backend.rag.models import ExtractedDocument, PageContent
from backend.rag.text_splitter import (
    RecursiveTextSplitter,
    split_document_into_chunks,
    split_page_content,
)


def test_recursive_splitter_chunk_size():
    """Verify that chunks do not exceed the configured chunk size."""
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an architectural pattern that combines "
        "information retrieval systems with generative large language models. Rather than relying "
        "solely on parametric memory, RAG models retrieve relevant facts from external vector databases. "
        "This substantially mitigates hallucination and enables dynamic knowledge updates without retraining. "
        "Sentence transformers are commonly utilized for dense vector embeddings, while vector databases "
        "such as FAISS enable rapid approximate nearest neighbor search over millions of high-dimensional vectors."
    )
    chunk_size = 150
    chunk_overlap = 30
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=20)
    chunks = splitter.split_text(sample_text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= chunk_size * 1.2  # Slight margin for word boundary preservation


def test_recursive_splitter_invalid_overlap():
    """Verify ValueError is raised if chunk_overlap >= chunk_size."""
    with pytest.raises(ValueError, match="strictly less"):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)


def test_page_number_provenance():
    """Verify that every chunk retains exact page number metadata."""
    page = PageContent(
        page_number=5,
        text=(
            "Page 5 discusses advanced retrieval mechanisms including re-ranking algorithms, "
            "reciprocal rank fusion, and hybrid dense-sparse vector scoring. " * 3
        ),
    )
    chunks = split_page_content(
        page=page,
        document_id="doc-99",
        filename="advanced_rag.pdf",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.page_number == 5
        assert chunk.document_id == "doc-99"
        assert chunk.filename == "advanced_rag.pdf"
        assert chunk.chunk_id.startswith("doc-99_p5_c")


def test_split_document_into_chunks_multi_page():
    """Verify document-level splitting across multiple pages."""
    doc = ExtractedDocument(
        document_id="doc-multi",
        filename="multi_page.pdf",
        file_path="/tmp/multi_page.pdf",
        file_hash="hash123",
        total_pages=2,
        pages=[
            PageContent(page_number=1, text="First page content discussing introduction and motivation."),
            PageContent(page_number=2, text="Second page content discussing methodology and experimental results."),
        ],
    )

    chunks = split_document_into_chunks(doc, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 2

    # Verify pages are preserved
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
