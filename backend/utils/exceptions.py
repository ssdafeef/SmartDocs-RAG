"""Custom domain exceptions for the SmartDocs RAG pipeline.

Provides granular, descriptive error types for predictable error handling
and clean API response mappings.
"""


class SmartDocsException(Exception):
    """Base exception for all application-level errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PDFProcessingError(SmartDocsException):
    """Raised when PDF extraction, parsing, or loading fails."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message=f"PDF Processing Error: {message}", status_code=status_code)


class TextChunkingError(SmartDocsException):
    """Raised when chunking or text splitting operations fail."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message=f"Text Chunking Error: {message}", status_code=status_code)


class EmbeddingGenerationError(SmartDocsException):
    """Raised when Sentence Transformer model fails to encode text."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message=f"Embedding Error: {message}", status_code=status_code)


class VectorStoreError(SmartDocsException):
    """Raised when FAISS indexing, persistence, or search fails."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message=f"Vector Store Error: {message}", status_code=status_code)


class DocumentNotFoundError(SmartDocsException):
    """Raised when an operation is attempted on an unknown document ID."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' not found.",
            status_code=404,
        )


class LLMGenerationError(SmartDocsException):
    """Raised when Gemini API request fails or encounters rate limits."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message=f"LLM Generation Error: {message}", status_code=status_code)
