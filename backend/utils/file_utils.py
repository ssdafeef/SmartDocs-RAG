"""File utilities for validation, hashing, and filename sanitization.

Provides robust helpers to manage document uploads and deduplication safely.
"""

import hashlib
import re
from pathlib import Path
from typing import Union


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hash of a file for content verification and deduplication.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """Compute the SHA-256 hash of raw bytes.

    Args:
        data: Raw byte content.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize an uploaded filename to prevent directory traversal or invalid characters.

    Args:
        filename: Original file name string.

    Returns:
        Safe, sanitized filename string.
    """
    # Strip any directory path components
    basename = Path(filename).name

    # Remove non-alphanumeric characters except dots, hyphens, and underscores
    clean_name = re.sub(r"[^\w\s\.-]", "", basename).strip()

    # Collapse consecutive whitespace or dots
    clean_name = re.sub(r"[\s]+", "_", clean_name)
    clean_name = re.sub(r"\.{2,}", ".", clean_name)

    return clean_name or "uploaded_document.pdf"


def format_file_size(num_bytes: int) -> str:
    """Format byte count into human-readable size string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., '1.25 MB', '450 KB').
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}".replace(".00", "")
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"
