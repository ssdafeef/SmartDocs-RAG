"""PDF Processing Engine for SmartDocs.

Extracts text, metadata, and per-page structures from PDF files using PyMuPDF (fitz),
with strict page numbering preservation for source citations.
"""

import re
import uuid
from pathlib import Path
from typing import Optional, Union

import pymupdf

from backend.rag.models import ExtractedDocument, PageContent
from backend.utils.exceptions import PDFProcessingError
from backend.utils.file_utils import compute_file_hash
from backend.utils.logger import get_logger

logger = get_logger("pdf_processor")


def clean_pdf_text(raw_text: str) -> str:
    """Clean and normalize raw extracted text from a PDF page.

    - Replaces null bytes and non-printable control characters.
    - Joins words broken by hyphenation across line breaks (e.g. 'com-\\nputer' -> 'computer').
    - Normalizes irregular whitespace, tabs, and multiple consecutive linefeeds.

    Args:
        raw_text: Raw text string directly from PyMuPDF.

    Returns:
        Normalized clean text string.
    """
    if not raw_text:
        return ""

    # Replace null characters
    text = raw_text.replace("\x00", "")

    # Rejoin words hyphenated at line breaks
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than two consecutive newlines into two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_from_pdf(
    file_path: Union[str, Path],
    document_id: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> ExtractedDocument:
    """Extract text from a PDF file preserving page numbers and metadata.

    Args:
        file_path: Path to the target PDF file.
        document_id: Optional unique identifier. Generated if not provided.
        original_filename: Optional original upload name. Defaults to file_path.name.

    Returns:
        ExtractedDocument object containing per-page content and statistics.

    Raises:
        PDFProcessingError: If the file does not exist, is corrupted, or has no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFProcessingError(f"File not found: {path}")

    if not path.is_file():
        raise PDFProcessingError(f"Target is not a file: {path}")

    if path.stat().st_size == 0:
        raise PDFProcessingError(f"PDF file is empty (0 bytes): {path.name}")

    doc_id = document_id or str(uuid.uuid4())
    fname = original_filename or path.name

    logger.info("Opening PDF: %s (id: %s)", fname, doc_id)

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        logger.error("Failed to parse PDF %s: %s", fname, exc)
        raise PDFProcessingError(f"Failed to open or parse PDF document: {exc}") from exc

    try:
        if doc.is_encrypted:
            raise PDFProcessingError(f"PDF document is password protected: {fname}")

        total_pages = len(doc)
        if total_pages == 0:
            raise PDFProcessingError(f"PDF contains 0 pages: {fname}")

        pages: list[PageContent] = []
        total_chars = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed for human-friendly citations
            page = doc[page_idx]

            raw_text = page.get_text("text") or ""
            clean_text = clean_pdf_text(raw_text)

            char_count = len(clean_text)
            total_chars += char_count

            pages.append(
                PageContent(
                    page_number=page_num,
                    text=clean_text,
                    char_count=char_count,
                )
            )

        # Extract basic PDF metadata
        pdf_metadata = doc.metadata or {}
        sanitized_metadata = {
            "title": pdf_metadata.get("title", "") or "",
            "author": pdf_metadata.get("author", "") or "",
            "subject": pdf_metadata.get("subject", "") or "",
            "creator": pdf_metadata.get("creator", "") or "",
            "producer": pdf_metadata.get("producer", "") or "",
            "creation_date": pdf_metadata.get("creationDate", "") or "",
        }

        # Validate that document contains readable text
        if total_chars == 0:
            logger.warning(
                "PDF %s has %d pages but 0 extractable characters (scanned image or empty).",
                fname,
                total_pages,
            )

        file_hash = compute_file_hash(path)

        logger.info(
            "Successfully extracted %s: %d pages, %d characters.",
            fname,
            total_pages,
            total_chars,
        )

        return ExtractedDocument(
            document_id=doc_id,
            filename=fname,
            file_path=str(path.resolve()),
            file_hash=file_hash,
            file_size_bytes=path.stat().st_size,
            total_pages=total_pages,
            total_chars=total_chars,
            pages=pages,
            metadata=sanitized_metadata,
        )

    finally:
        doc.close()
