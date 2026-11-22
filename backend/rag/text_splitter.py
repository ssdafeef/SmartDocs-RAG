"""Text Splitting and Semantic Chunking Module for SmartDocs.

Splits extracted document pages into overlapping text chunks while strictly
preserving source page numbers and document metadata for accurate citations.
"""

import re
from typing import List, Optional

from backend.config import settings
from backend.rag.models import ExtractedDocument, PageContent, TextChunk
from backend.utils.logger import get_logger

logger = get_logger("text_splitter")


class RecursiveTextSplitter:
    """Recursively splits text using hierarchical separators to preserve semantic context."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ) -> None:
        """Initialize splitter with chunk size, overlap, and separator hierarchy.

        Args:
            chunk_size: Maximum characters per chunk (default from settings).
            chunk_overlap: Number of characters shared between adjacent chunks.
            min_chunk_size: Minimum characters required to retain a chunk.
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or settings.MIN_CHUNK_SIZE

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})."
            )

        # Hierarchy of separators: paragraphs, lines, sentences, clauses, words
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """Split text by the highest-level separator that produces pieces."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)
        pieces: List[str] = []

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) <= self.chunk_size:
                pieces.append(s_clean)
            else:
                # Recurse with lower-level separators
                deeper = self._split_text_with_separators(s_clean, remaining_separators)
                pieces.extend(deeper)

        return pieces

    def split_text(self, text: str) -> List[str]:
        """Split a string into overlapping chunks.

        Args:
            text: Input text string.

        Returns:
            List of chunk strings respecting chunk_size and overlap.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned] if len(cleaned) >= self.min_chunk_size else []

        pieces = self._split_text_with_separators(cleaned, self.separators)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_length + piece_len + 1 > self.chunk_size and current_chunk:
                # Assemble current chunk
                combined = " ".join(current_chunk).strip()
                if len(combined) >= self.min_chunk_size:
                    chunks.append(combined)

                # Form overlap from trailing pieces
                overlap_accum: List[str] = []
                overlap_len = 0
                for prev_piece in reversed(current_chunk):
                    if overlap_len + len(prev_piece) <= self.chunk_overlap:
                        overlap_accum.insert(0, prev_piece)
                        overlap_len += len(prev_piece) + 1
                    else:
                        break

                current_chunk = overlap_accum
                current_length = sum(len(p) + 1 for p in current_chunk)

            current_chunk.append(piece)
            current_length += piece_len + 1

        if current_chunk:
            final_chunk = " ".join(current_chunk).strip()
            if len(final_chunk) >= self.min_chunk_size:
                chunks.append(final_chunk)

        return chunks


def split_page_content(
    page: PageContent,
    document_id: str,
    filename: str,
    start_chunk_idx: int = 0,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """Split text on a single page into TextChunks with page-level provenance.

    Args:
        page: PageContent object.
        document_id: Unique document identifier.
        filename: Original PDF filename.
        start_chunk_idx: Starting sequential index for chunks.
        chunk_size: Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.

    Returns:
        List of TextChunk models.
    """
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_text(page.text)

    chunks: List[TextChunk] = []
    char_offset = 0

    for i, chunk_text in enumerate(text_chunks):
        chunk_idx = start_chunk_idx + i
        chunk_id = f"{document_id}_p{page.page_number}_c{chunk_idx}"

        # Locate character offset on page if possible
        pos = page.text.find(chunk_text[:50], char_offset)
        if pos != -1:
            char_start = pos
            char_end = pos + len(chunk_text)
            char_offset = pos + 1
        else:
            char_start = 0
            char_end = len(chunk_text)

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page_number=page.page_number,
                chunk_index=chunk_idx,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                word_count=len(chunk_text.split()),
            )
        )

    return chunks


def split_document_into_chunks(
    document: ExtractedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """Split an entire ExtractedDocument into TextChunks across all pages.

    Args:
        document: ExtractedDocument model with per-page text.
        chunk_size: Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.

    Returns:
        List of all TextChunk models in the document.
    """
    all_chunks: List[TextChunk] = []
    chunk_counter = 0

    for page in document.pages:
        page_chunks = split_page_content(
            page=page,
            document_id=document.document_id,
            filename=document.filename,
            start_chunk_idx=chunk_counter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)
        chunk_counter += len(page_chunks)

    logger.info(
        "Chunked document %s (%d pages) into %d chunks (size=%d, overlap=%d).",
        document.filename,
        document.total_pages,
        len(all_chunks),
        chunk_size or settings.CHUNK_SIZE,
        chunk_overlap or settings.CHUNK_OVERLAP,
    )

    return all_chunks
