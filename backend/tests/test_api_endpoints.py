"""Direct unit tests for FastAPI route handler functions using asyncio.run."""

import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app, root
from backend.routes.chat import chat
from backend.routes.documents import clear_all_documents, delete_document, list_documents
from backend.routes.health import health_check
from backend.routes.schemas import ChatRequest

client = TestClient(app)


def test_root_endpoint_direct():
    """Verify root / returns application metadata."""
    res = asyncio.run(root())
    assert res["status"] == "online"
    assert "SmartDocs" in res["application"]
    assert "endpoints" in res


def test_health_endpoint_direct():
    """Verify health_check returns healthy status and system metrics."""
    res = asyncio.run(health_check())
    assert res.status == "healthy"
    assert res.version == "1.0.0"
    assert res.total_documents >= 0
    assert res.total_vectors >= 0


def test_list_documents_direct():
    """Verify list_documents returns list structure."""
    res = asyncio.run(list_documents())
    assert isinstance(res.documents, list)
    assert res.total_count == len(res.documents)


def test_chat_direct_no_documents(monkeypatch):
    """Verify chat route responds when vector store has no documents."""
    from backend.rag.rag_pipeline import rag_pipeline

    class EmptyStore:
        def get_stats(self):
            return {"total_vectors": 0}

    monkeypatch.setattr(rag_pipeline, "vector_store", EmptyStore())
    req = ChatRequest(query="What is artificial intelligence?")
    res = asyncio.run(chat(req))
    assert "No documents have been uploaded yet" in res.answer
    assert res.found_relevant_context is False
    assert len(res.citations) == 0


def test_delete_nonexistent_document_direct():
    """Verify delete_document raises 404 HTTPException for unknown ID."""
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(delete_document("nonexistent-id-999"))
    assert excinfo.value.status_code == 404


def test_clear_all_documents_direct(monkeypatch):
    """Verify clear-all reporting without deleting a developer's local library."""
    from backend.routes import documents as documents_route

    class EmptyStore:
        def list_documents(self):
            return []

        def clear_all(self):
            self.cleared = True

    store = EmptyStore()
    monkeypatch.setattr(documents_route, "vector_store", store)
    res = asyncio.run(clear_all_documents())
    assert "cleared" in res.message.lower()
    assert res.cleared_documents_count >= 0
    assert store.cleared


def test_http_routes_and_request_validation():
    """Exercise the ASGI app and validation layer, not only Python handlers."""
    assert client.get("/").status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert client.get("/docs").status_code == 200

    invalid_chat = client.post("/api/chat", json={"query": ""})
    assert invalid_chat.status_code == 422


def test_http_chat_uses_empty_library_response(monkeypatch):
    """Verify the public JSON chat contract for a library without documents."""
    from backend.rag.rag_pipeline import rag_pipeline

    class EmptyStore:
        def get_stats(self):
            return {"total_vectors": 0}

    monkeypatch.setattr(rag_pipeline, "vector_store", EmptyStore())
    response = client.post("/api/chat", json={"query": "What is in my library?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["found_relevant_context"] is False
    assert payload["citations"] == []
    assert payload["session_id"]
