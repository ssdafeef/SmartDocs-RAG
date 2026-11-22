"""PDF Document Upload and Processing Route for SmartDocs."""

import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.config import settings
from backend.rag.embeddings import embedding_manager
from backend.rag.pdf_processor import extract_text_from_pdf
from backend.rag.text_splitter import split_document_into_chunks
from backend.rag.vector_store import vector_store
from backend.routes.schemas import UploadFileResult, UploadResponse
from backend.utils.exceptions import PDFProcessingError
from backend.utils.file_utils import sanitize_filename
from backend.utils.logger import get_logger

logger = get_logger("routes_upload")
router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "",
    response_model=UploadResponse,
    summary="Upload and process one or more PDF documents",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more PDF documents to upload"),
) -> UploadResponse:
    """Accept PDF uploads, extract text per page, chunk, compute embeddings, and index into FAISS."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were attached to the upload request.",
        )

    results: List[UploadFileResult] = []
    total_indexed_chunks = 0

    for file in files:
        raw_filename = file.filename or "document.pdf"
        clean_name = sanitize_filename(raw_filename)

        logger.info("Processing upload file: %s (sanitized: %s)", raw_filename, clean_name)

        # Validate PDF extension
        if not clean_name.lower().endswith(".pdf"):
            results.append(
                UploadFileResult(
                    document_id="",
                    filename=raw_filename,
                    pages=0,
                    chunks=0,
                    status="failed",
                    message="Invalid file format. Only PDF files are supported.",
                )
            )
            continue

        doc_id = str(uuid.uuid4())
        dest_filename = f"{doc_id[:8]}_{clean_name}"
        dest_path = settings.UPLOADS_DIR / dest_filename

        try:
            # Read file bytes and check size
            content = await file.read()
            if len(content) == 0:
                results.append(
                    UploadFileResult(
                        document_id=doc_id,
                        filename=raw_filename,
                        pages=0,
                        chunks=0,
                        status="failed",
                        message="File is empty (0 bytes).",
                    )
                )
                continue

            max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            if len(content) > max_bytes:
                results.append(
                    UploadFileResult(
                        document_id=doc_id,
                        filename=raw_filename,
                        pages=0,
                        chunks=0,
                        status="failed",
                        message=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB.",
                    )
                )
                continue

            # Save uploaded PDF to storage
            with open(dest_path, "wb") as f:
                f.write(content)

            # 1. Extract text and pages with PyMuPDF
            extracted_doc = extract_text_from_pdf(
                file_path=dest_path,
                document_id=doc_id,
                original_filename=raw_filename,
            )

            # 2. Chunk text with page preservation
            chunks = split_document_into_chunks(extracted_doc)

            # 3. Generate embeddings and index in FAISS
            if chunks:
                embeddings = embedding_manager.embed_chunks(chunks)
                vector_store.add_document_chunks(extracted_doc, chunks, embeddings)
                total_indexed_chunks += len(chunks)
            else:
                vector_store.add_document_chunks(extracted_doc, [], None)
                logger.warning("Document '%s' produced 0 chunks.", raw_filename)

            if extracted_doc.total_chars == 0:
                result_message = (
                    "No selectable text was found. This PDF may be scanned; run OCR before uploading."
                )
            else:
                result_message = f"Indexed {len(chunks)} chunks across {extracted_doc.total_pages} pages."

            results.append(
                UploadFileResult(
                    document_id=doc_id,
                    filename=raw_filename,
                    pages=extracted_doc.total_pages,
                    chunks=len(chunks),
                    status="success",
                    message=result_message,
                )
            )

        except PDFProcessingError as exc:
            logger.error("PDF processing failed for %s: %s", raw_filename, exc)
            if dest_path.exists():
                dest_path.unlink()
            results.append(
                UploadFileResult(
                    document_id=doc_id,
                    filename=raw_filename,
                    pages=0,
                    chunks=0,
                    status="failed",
                    message=exc.message,
                )
            )
        except Exception as exc:
            logger.error("Unexpected error indexing %s: %s", raw_filename, exc)
            if dest_path.exists():
                dest_path.unlink()
            results.append(
                UploadFileResult(
                    document_id=doc_id,
                    filename=raw_filename,
                    pages=0,
                    chunks=0,
                    status="failed",
                    message=f"Indexing error: {exc}",
                )
            )

    successful = sum(1 for r in results if r.status == "success")
    summary = f"Processed {len(files)} file(s): {successful} successfully indexed into FAISS."

    return UploadResponse(
        message=summary,
        uploaded_files=results,
        total_indexed_chunks=total_indexed_chunks,
    )
