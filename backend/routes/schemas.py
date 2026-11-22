"""Pydantic request and response schemas for SmartDocs REST API."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.rag.models import Citation


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Application status")
    version: str = Field(..., description="Application version")
    total_documents: int = Field(default=0, description="Total indexed documents")
    total_vectors: int = Field(default=0, description="Total vectors in FAISS index")
    gemini_configured: bool = Field(default=False, description="Whether Gemini API key is configured")


class DocumentItem(BaseModel):
    """Information for a single indexed document."""

    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Document file name")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    total_pages: int = Field(default=0, description="Page count")
    total_chars: int = Field(default=0, description="Character count")
    chunk_count: int = Field(default=0, description="Number of text chunks")
    upload_date: str = Field(..., description="ISO upload timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="PDF metadata")


class DocumentListResponse(BaseModel):
    """List of all indexed documents."""

    documents: List[DocumentItem] = Field(default_factory=list)
    total_count: int = Field(default=0)


class UploadFileResult(BaseModel):
    """Result of uploading a single document."""

    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str = "success"
    message: Optional[str] = None


class UploadResponse(BaseModel):
    """Overall response for document upload endpoint."""

    message: str
    uploaded_files: List[UploadFileResult] = Field(default_factory=list)
    total_indexed_chunks: int = 0


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document."""

    message: str
    document_id: str
    remaining_documents: int


class ClearAllResponse(BaseModel):
    """Response after wiping all documents."""

    message: str
    cleared_documents_count: int


class ChatRequest(BaseModel):
    """Request payload for /chat endpoint."""

    query: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(default=None, description="Optional conversational session ID")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold",
    )


class ChatResponse(BaseModel):
    """Response payload for /chat endpoint."""

    answer: str = Field(..., description="Generated answer based on documents")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    found_relevant_context: bool = Field(default=True, description="Whether relevant context answered query")
    retrieved_chunks_count: int = Field(default=0, description="Number of context chunks retrieved")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    detail: str = Field(..., description="Error message description")
    error_type: Optional[str] = Field(default=None, description="Exception classification")
