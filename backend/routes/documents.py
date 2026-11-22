"""Document management API routes for SmartDocs."""

from typing import List
from fastapi import APIRouter, HTTPException, Path, status

from backend.rag.vector_store import vector_store
from backend.routes.schemas import (
    ClearAllResponse,
    DeleteDocumentResponse,
    DocumentItem,
    DocumentListResponse,
)
from backend.utils.exceptions import DocumentNotFoundError
from backend.utils.logger import get_logger

logger = get_logger("routes_documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded and indexed documents",
)
async def list_documents() -> DocumentListResponse:
    """Retrieve all documents currently indexed in the FAISS vector database."""
    docs = vector_store.list_documents()
    items = [
        DocumentItem(
            document_id=d["document_id"],
            filename=d["filename"],
            file_size_bytes=d.get("file_size_bytes", 0),
            total_pages=d.get("total_pages", 0),
            total_chars=d.get("total_chars", 0),
            chunk_count=d.get("chunk_count", 0),
            upload_date=d.get("upload_date", ""),
            metadata=d.get("metadata", {}),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items, total_count=len(items))


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a single document and update vector index",
)
async def delete_document(
    document_id: str = Path(..., description="UUID of the document to delete"),
) -> DeleteDocumentResponse:
    """Remove a document from the system and re-index the remaining document chunks."""
    try:
        vector_store.delete_document(document_id)
        remaining = len(vector_store.list_documents())
        return DeleteDocumentResponse(
            message=f"Document '{document_id}' successfully removed and vector index rebuilt.",
            document_id=document_id,
            remaining_documents=remaining,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {exc}",
        ) from exc


@router.delete(
    "",
    response_model=ClearAllResponse,
    summary="Remove all documents and reset vector index",
)
async def clear_all_documents() -> ClearAllResponse:
    """Clear all uploaded PDF files and reset the FAISS index to an empty state."""
    count = len(vector_store.list_documents())
    vector_store.clear_all()
    return ClearAllResponse(
        message="All uploaded documents and index entries were successfully cleared.",
        cleared_documents_count=count,
    )
