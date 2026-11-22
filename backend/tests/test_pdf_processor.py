"""Unit tests for the PDF processing engine."""

import tempfile
from pathlib import Path
import pytest
import pymupdf

from backend.rag.pdf_processor import clean_pdf_text, extract_text_from_pdf
from backend.utils.exceptions import PDFProcessingError


def test_clean_pdf_text():
    """Verify clean_pdf_text sanitizes raw PDF strings accurately."""
    raw = "This is a test of hyphen-\nation in doc-\numents.\x00 Multiple   spaces   here.\n\n\n\nParagraph 2."
    cleaned = clean_pdf_text(raw)
    assert "hyphenation" in cleaned
    assert "documents" in cleaned
    assert "\x00" not in cleaned
    assert "Multiple spaces here." in cleaned
    assert "\n\nParagraph 2." in cleaned


def test_extract_text_from_valid_pdf():
    """Create a temporary multi-page PDF and verify per-page extraction."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Create a synthetic 2-page PDF with PyMuPDF
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "SmartDocs Retrieval-Augmented Generation Architecture Overview.")

        p2 = doc.new_page()
        p2.insert_text((50, 72), "Vector embeddings are generated using all-MiniLM-L6-v2 transformer model.")
        doc.save(str(tmp_path))
        doc.close()

        extracted = extract_text_from_pdf(tmp_path, document_id="doc-test-123")

        assert extracted.document_id == "doc-test-123"
        assert extracted.total_pages == 2
        assert len(extracted.pages) == 2

        # Check page 1
        assert extracted.pages[0].page_number == 1
        assert "SmartDocs" in extracted.pages[0].text

        # Check page 2
        assert extracted.pages[1].page_number == 2
        assert "all-MiniLM-L6-v2" in extracted.pages[1].text
        assert extracted.total_chars > 0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_extract_text_from_nonexistent_file():
    """Verify PDFProcessingError is raised when file does not exist."""
    with pytest.raises(PDFProcessingError, match="File not found"):
        extract_text_from_pdf("nonexistent_path_98765.pdf")


def test_extract_text_from_empty_file():
    """Verify PDFProcessingError is raised for zero-byte files."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(PDFProcessingError, match="empty"):
            extract_text_from_pdf(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
