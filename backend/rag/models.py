"""Domain data structures and models for the SmartDocs RAG pipeline."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Represents extracted text content from a single PDF page."""

    page_number: int = Field(..., description="1-indexed page number from the PDF")
    text: str = Field(..., description="Cleaned text extracted from this page")
    char_count: int = Field(default=0, description="Number of characters on this page")


class ExtractedDocument(BaseModel):
    """Represents a fully processed PDF document with per-page content."""

    document_id: str = Field(..., description="Unique UUID identifier for this document")
    filename: str = Field(..., description="Original uploaded filename")
    file_path: str = Field(..., description="Absolute path on disk")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    total_pages: int = Field(default=0, description="Total number of pages")
    total_chars: int = Field(default=0, description="Total extracted character count")
    pages: List[PageContent] = Field(default_factory=list, description="List of per-page content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted PDF metadata")


class TextChunk(BaseModel):
    """Represents a discrete text chunk produced by the chunking engine."""

    chunk_id: str = Field(..., description="Unique identifier: doc_id_p{page}_{index}")
    document_id: str = Field(..., description="Associated document ID")
    filename: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="Source page number (1-indexed)")
    chunk_index: int = Field(..., description="Sequential chunk index within document")
    text: str = Field(..., description="The chunk text payload")
    char_start: int = Field(default=0, description="Character start offset on the page")
    char_end: int = Field(default=0, description="Character end offset on the page")
    word_count: int = Field(default=0, description="Word count in chunk")


class Citation(BaseModel):
    """Source citation displayed alongside answers for verification."""

    citation_index: int = Field(..., description="1-based citation index (e.g. [1])")
    document_id: str = Field(..., description="ID of source document")
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number where information originates")
    snippet: str = Field(..., description="Extracted text snippet from source")
    relevance_score: float = Field(default=0.0, description="Cosine similarity score (0.0 to 1.0)")


class SearchResult(BaseModel):
    """Result of a semantic search query against the vector store."""

    chunk: TextChunk = Field(..., description="Matched text chunk")
    score: float = Field(..., description="Similarity score (higher is more similar)")


class RAGResponse(BaseModel):
    """Complete answer produced by the RAG pipeline."""

    answer: str = Field(..., description="Synthesized answer strictly based on retrieved sources")
    citations: List[Citation] = Field(default_factory=list, description="Verified source citations")
    found_relevant_context: bool = Field(default=True, description="Whether relevant documents answered query")
    retrieved_chunks_count: int = Field(default=0, description="Number of context chunks passed to LLM")
