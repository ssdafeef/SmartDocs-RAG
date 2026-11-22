"""Unit tests for backend configuration settings."""

import pytest
from backend.config import settings


def test_settings_initialization():
    """Verify settings defaults are populated correctly."""
    assert settings.PROJECT_NAME == "SmartDocs - AI Document Research Assistant"
    assert settings.EMBEDDING_DIMENSION == 384
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
    assert settings.TOP_K_RETRIEVAL >= 1


def test_storage_directories_exist():
    """Verify upload and vector store directories are created automatically."""
    assert settings.STORAGE_DIR.exists()
    assert settings.UPLOADS_DIR.exists()
    assert settings.VECTOR_STORE_DIR.exists()


def test_allowed_origins():
    """Verify CORS origins include development localhost ports."""
    assert "http://localhost:5173" in settings.ALLOWED_ORIGINS
    assert "http://127.0.0.1:5173" in settings.ALLOWED_ORIGINS
