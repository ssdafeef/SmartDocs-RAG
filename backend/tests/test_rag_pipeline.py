"""Unit tests for the RAG pipeline orchestrator and context construction."""

import pytest
from backend.rag.models import SearchResult, TextChunk
from backend.rag.rag_pipeline import RAGPipeline, SessionHistoryManager
from backend.rag.vector_store import FAISSVectorStore


def test_session_history_manager():
    """Verify conversational session turns are tracked and trimmed properly."""
    manager = SessionHistoryManager(max_history_turns=2)
    session_id = "test-session-1"

    manager.add_turn(session_id, "user", "Hello")
    manager.add_turn(session_id, "assistant", "Hi there!")
    history = manager.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # Clear session
    manager.clear_session(session_id)
    assert len(manager.get_history(session_id)) == 0


def test_build_context():
    """Verify build_context formats excerpts with citations headers and creates Citation models."""
    pipeline = RAGPipeline()

    chunk1 = TextChunk(
        chunk_id="c1",
        document_id="doc1",
        filename="report.pdf",
        page_number=3,
        chunk_index=0,
        text="The company revenue increased by 25% in Q3.",
    )
    chunk2 = TextChunk(
        chunk_id="c2",
        document_id="doc1",
        filename="report.pdf",
        page_number=4,
        chunk_index=1,
        text="Operating margins expanded to 18.5%.",
    )

    results = [
        SearchResult(chunk=chunk1, score=0.88),
        SearchResult(chunk=chunk2, score=0.75),
    ]

    context_str, citations = pipeline.build_context(results)

    assert "[Source 1: report.pdf | Page 3]" in context_str
    assert "[Source 2: report.pdf | Page 4]" in context_str
    assert "revenue increased by 25%" in context_str

    assert len(citations) == 2
    assert citations[0].citation_index == 1
    assert citations[0].page_number == 3
    assert citations[0].filename == "report.pdf"
    assert citations[0].relevance_score == 0.88

    assert citations[1].citation_index == 2
    assert citations[1].page_number == 4


def test_generate_answer_empty_query():
    """Verify empty query returns prompt to enter question."""
    pipeline = RAGPipeline()
    response = pipeline.generate_answer("")
    assert "enter a question" in response.answer
    assert response.found_relevant_context is False


def test_generate_answer_empty_vector_store():
    """Verify empty vector store instructs user to upload PDF files."""
    empty_store = FAISSVectorStore(dimension=4)
    empty_store.clear_all()
    pipeline = RAGPipeline(v_store=empty_store)

    response = pipeline.generate_answer("What is the quarterly revenue?")
    assert "No documents have been uploaded yet" in response.answer
    assert response.found_relevant_context is False
