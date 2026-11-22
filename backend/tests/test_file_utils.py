"""Unit tests for file utility functions."""

import tempfile
from pathlib import Path
from backend.utils.file_utils import (
    compute_bytes_hash,
    compute_file_hash,
    format_file_size,
    sanitize_filename,
)


def test_sanitize_filename_removes_dangerous_characters():
    """Verify directory traversal and invalid characters are sanitized."""
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("test*file<name>?.pdf") == "testfilename.pdf"
    assert sanitize_filename("my report  final (2026).pdf") == "my_report_final_2026.pdf"
    assert sanitize_filename("") == "uploaded_document.pdf"


def test_format_file_size():
    """Verify byte count conversion into human-readable units."""
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1 KB"
    assert format_file_size(1048576) == "1 MB"
    assert format_file_size(1572864) == "1.5 MB"


def test_compute_bytes_hash():
    """Verify SHA-256 hash generation for raw bytes."""
    data = b"SmartDocs RAG Engine Test"
    hash1 = compute_bytes_hash(data)
    hash2 = compute_bytes_hash(data)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_compute_file_hash():
    """Verify file hash computation matches byte hash."""
    data = b"Temporary test document content for hashing"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        file_hash = compute_file_hash(tmp_path)
        bytes_hash = compute_bytes_hash(data)
        assert file_hash == bytes_hash
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
